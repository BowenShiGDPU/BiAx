#!/usr/bin/env python3
"""BORA herb-drug model.

Herb constituents are encoded as a masked token set. Drug context is encoded
from semantic, graph and structure features. Query-conditioned readout and
observed-relation retrieval provide complementary prediction channels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .unseen_herb_expert import (UnseenHerbExpert,
                                 routed_probability_torch)

EPS = 1e-6


@dataclass(frozen=True)
class Config:
    semantic_dim: int = 1024
    graph_dim: int = 256
    structure_dim: int = 2049
    constituent_dim: int = 2049
    pair_mechanism_dim: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    memory_smoothing: float = 5.0
    max_constituents: int = 64
    memory_topk_herb: int = 20
    memory_topk_drug: int = 20
    memory_tau_init: float = -1.0
    support_dropout: float = 0.3
    support_dropout_stop_epoch: int = 0   # 0 keeps it on for every epoch
    support_dropout_drug: float = 0.0     # the same, drawn per drug column
    fusion: str = "A"
    no_support_bias: bool = False
    cold_expert: bool = False
    hard_cold_route: bool = False
    graph_expert: bool = False
    dropout: float = 0.2


class ChannelEncoder(nn.Module):
    """The three feature blocks as three tokens through a small transformer."""

    def __init__(self, dimensions: tuple[int, ...], cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(size), nn.Linear(size, d), nn.GELU(),
                          nn.Dropout(cfg.dropout))
            for size in dimensions
        ])
        self.channel_embedding = nn.Parameter(torch.zeros(len(dimensions), d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=2 * d,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, *values: torch.Tensor):
        tokens = torch.stack(
            [p(v) for p, v in zip(self.projections, values)], dim=1)
        tokens = self.norm(self.blocks(tokens + self.channel_embedding.unsqueeze(0)))
        return tokens, tokens.mean(dim=1)


class ConstituentAttention(nn.Module):
    """Masked self-attention over a herb's constituents, after the formula arm's
    MaterialSelfAttention: a padded set with an attention mask."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model
        self.attention = nn.MultiheadAttention(d, cfg.n_heads, cfg.dropout,
                                               batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        pad = ~mask
        pad = pad & (~pad.all(dim=1, keepdim=True))   # never mask a whole row
        a, _ = self.attention(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))


class ConditionalReadout(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                               cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, condition: torch.Tensor, tokens: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        pad = None
        if mask is not None:
            pad = ~mask
            pad = pad & (~pad.all(dim=1, keepdim=True))
        value, _ = self.attention(condition.unsqueeze(1), tokens, tokens,
                                  key_padding_mask=pad, need_weights=False)
        return self.norm(condition + value.squeeze(1))


class BORAHerbDrug(nn.Module):
    def __init__(self, n_herb: int, n_drug: int, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        d = c.d_model
        self.drug_encoder = ChannelEncoder(
            (c.semantic_dim, c.graph_dim, c.structure_dim), c)
        self.herb_encoder = ChannelEncoder(
            (c.semantic_dim, c.graph_dim, c.structure_dim), c)
        self.constituent_projection = nn.Sequential(
            nn.LayerNorm(c.constituent_dim), nn.Linear(c.constituent_dim, d),
            nn.GELU(), nn.Dropout(c.dropout))
        self.constituent_blocks = nn.ModuleList(
            [ConstituentAttention(c) for _ in range(c.n_layers)])
        self.constituent_gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.readout = ConditionalReadout(c)
        self.herb_mix = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d),
                                      nn.GELU())
        self.structural = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(d, 1))
        self.mechanism = nn.Sequential(
            nn.LayerNorm(c.pair_mechanism_dim),
            nn.Linear(c.pair_mechanism_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.memory_herb = nn.Linear(d, d, bias=False)
        self.memory_drug = nn.Linear(d, d, bias=False)
        self.log_tau_herb = nn.Parameter(torch.tensor(c.memory_tau_init))
        self.log_tau_drug = nn.Parameter(torch.tensor(c.memory_tau_init))
        # present in the reference configuration, where it was never wired up; it is
        # kept unconditionally so that the reference configuration keeps the
        # parameter list it always had
        self.label_gate = nn.Sequential(
            nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))
        self.memory_scale = nn.Parameter(torch.ones(()))
        self.memory_bias = nn.Parameter(torch.zeros(()))
        # begins on the memory channel, as in the other task instances, where the
        # retrieval memory is already a strong standalone predictor
        self.channel_scale = nn.Parameter(torch.tensor(
            [1.0, 0.0, 1.0] if c.fusion in ("A", "C") else [0.0, 0.0, 1.0]))
        self.current_epoch = 0
        # what the memory channel should emit when a cell has no support at
        # all. Zero is not neutral: the channel sits at +2.3 on positives and
        # -0.3 on negatives, so an unsupported cell silently outranks a
        # supported negative. Starts at zero, so the default is unchanged.
        if c.no_support_bias:
            self.no_support_value = nn.Parameter(torch.zeros(()))
        # the feature-only expert, and its last layer at zero so that the
        # model starts exactly at the pathway it had before
        if c.cold_expert:
            raw = c.semantic_dim + c.graph_dim + c.structure_dim
            self.cold_head = nn.Sequential(
                nn.Linear(2 * raw, 512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 1))
            nn.init.zeros_(self.cold_head[-1].weight)
            nn.init.zeros_(self.cold_head[-1].bias)
        self.register_buffer("channel_balance", torch.ones(3))
        self.residual_weight = nn.Parameter(torch.zeros(n_drug, d))
        self.residual_bias = nn.Parameter(torch.zeros(n_drug))
        if c.graph_expert:
            self.graph_herb_update = nn.ModuleList(
                [nn.Linear(d, d) for _ in range(2)])
            self.graph_drug_update = nn.ModuleList(
                [nn.Linear(d, d) for _ in range(2)])
            self.graph_herb_norm = nn.ModuleList(
                [nn.LayerNorm(d) for _ in range(2)])
            self.graph_drug_norm = nn.ModuleList(
                [nn.LayerNorm(d) for _ in range(2)])
            self.graph_pair = nn.Sequential(
                nn.LayerNorm(4 * d), nn.Linear(4 * d, 2 * d), nn.GELU(),
                nn.Dropout(c.dropout), nn.Linear(2 * d, 1))
            self.graph_scale = nn.Parameter(torch.ones(()))

    # ------------------------------------------------------------------ herb
    def encode_herb(self, semantic, graph, structure, constituents, mask):
        """constituents: (H, S, constituent_dim); mask: (H, S) bool."""
        self._raw_herb = torch.cat([semantic, graph, structure], dim=-1)
        _, state = self.herb_encoder(semantic, graph, structure)
        x = self.constituent_projection(constituents)
        for block in self.constituent_blocks:
            x = block(x, mask)
        gate = torch.sigmoid(self.constituent_gate(x)).squeeze(-1) * mask
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (x * gate.unsqueeze(-1)).sum(dim=1) / denom
        return x, gate, self.herb_mix(torch.cat([state, pooled], dim=-1))

    def encode_drug(self, semantic, graph, structure):
        self._raw_drug = torch.cat([semantic, graph, structure], dim=-1)
        return self.drug_encoder(semantic, graph, structure)

    def route_unseen_herb(self, core_logit: torch.Tensor,
                          expert: UnseenHerbExpert,
                          herb_index: torch.Tensor,
                          drug_index: torch.Tensor,
                          alpha: float) -> torch.Tensor:
        """Apply the herb-support router after the core forward pass.

        The route is active only for herbs with no direct training support.
        Core and expert logits are converted to probabilities, clipped for
        numerical stability, and then composed in logit space, matching the
        released evaluation path.
        """
        if not hasattr(self, "_n_obs_herb"):
            raise RuntimeError("retrieve_relations must be evaluated before routing")
        route = self._n_obs_herb[herb_index] <= EPS
        core_probability = torch.sigmoid(core_logit)
        if not bool(route.any()):
            self.last_route = route.detach()
            return core_probability
        expert_logit = expert(self._raw_herb[herb_index],
                              self._raw_drug[drug_index])
        expert_probability = torch.sigmoid(expert_logit)
        self.last_route = route.detach()
        return routed_probability_torch(core_probability, expert_probability,
                                        route, alpha)

    # ---------------------------------------------------------------- memory
    @staticmethod
    def _sparse_softmax(logits: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or k >= logits.shape[-1]:
            return torch.softmax(logits, dim=-1)
        cut = logits.topk(k, dim=-1).values[..., -1:]
        return torch.softmax(logits.masked_fill(logits < cut, float("-inf")),
                             dim=-1)

    def retrieve_relations(self, herb_state, drug_state, labels, observed):
        """Two-sided diffusion of the observed label matrix, after the reference
        formula arm's memory operator.

        labels and observed are (n_herb, n_drug) and are built from the
        training fold only.
        """
        c = self.cfg
        stop = c.support_dropout_stop_epoch
        withhold = c.support_dropout > 0 and (
            stop <= 0 or self.current_epoch <= stop)
        if self.training and withhold:
            # a held-out entity arrives with its whole row of support missing,
            # so support is withheld by entity rather than by cell: dropping
            # individual cells only thins the support, it never reproduces the
            # condition the model actually meets at evaluation
            keep = (torch.rand(observed.shape[0], 1,
                               device=observed.device)
                    >= c.support_dropout).float()
            observed = observed * keep
            labels = labels * keep
        if self.training and c.support_dropout_drug > 0:
            # the same withholding along the other axis, so that a drug with
            # no support at all is a condition the model has actually met
            keep_d = (torch.rand(1, observed.shape[1],
                                 device=observed.device)
                      >= c.support_dropout_drug).float()
            observed = observed * keep_d
            labels = labels * keep_d
        self._n_obs_herb = observed.sum(dim=1)
        self._n_obs_drug = observed.sum(dim=0)
        uh = F.normalize(self.memory_herb(herb_state), dim=-1)
        ud = F.normalize(self.memory_drug(drug_state), dim=-1)
        Sh = self._sparse_softmax(
            uh @ uh.t() / self.log_tau_herb.exp().clamp(0.02, 10.0),
            c.memory_topk_herb)
        Sd = self._sparse_softmax(
            ud @ ud.t() / self.log_tau_drug.exp().clamp(0.02, 10.0),
            c.memory_topk_drug)
        numerator = Sh @ (observed * labels) @ Sd.t()
        denominator = Sh @ observed @ Sd.t()
        # a cell must not read its own label through the diagonal of either side
        self_h = torch.diagonal(Sh).unsqueeze(1)
        self_d = torch.diagonal(Sd).unsqueeze(0)
        joint = self_h * self_d * observed
        numerator = numerator - joint * labels
        denominator = denominator - joint
        prior = (observed * labels).sum() / observed.sum().clamp_min(EPS)
        # smoothing in units of the typical observed mass, so that the reference
        # prior-to-data ratio survives this arm's much sparser label matrix
        k = c.memory_smoothing * denominator.mean().clamp_min(EPS)
        table = (numerator + k * prior) / (denominator + k).clamp_min(EPS)
        if c.graph_expert:
            # The graph expert sees training-positive edges only. Its inputs
            # are detached so its transductive objective cannot rewrite the
            # inductive BORA encoders used for unseen entities.
            positive = (observed * labels).detach()
            gh, gd = herb_state.detach(), drug_state.detach()
            herb_degree = positive.sum(dim=1, keepdim=True).clamp_min(1.0)
            drug_degree = positive.sum(dim=0, keepdim=True).t().clamp_min(1.0)
            for hu, du, hn, dn in zip(
                    self.graph_herb_update, self.graph_drug_update,
                    self.graph_herb_norm, self.graph_drug_norm):
                h_neighbour = positive @ gd / herb_degree
                d_neighbour = positive.t() @ gh / drug_degree
                gh = hn(gh + hu(h_neighbour))
                gd = dn(gd + du(d_neighbour))
            self._graph_herb, self._graph_drug = gh, gd
        return table, denominator

    # ----------------------------------------------------------------- score
    def score(self, herb_tokens, herb_mask, herb_state, drug_state,
              herb_index, drug_index, pair_mechanism, memory_table,
              query_label=None, exclude_query=False):
        condition = drug_state[drug_index]
        # the drug conditions the herb's constituent set
        conditioned = self.readout(condition, herb_tokens[herb_index],
                                   herb_mask[herb_index])
        herb = herb_state[herb_index]
        structural = self.structural(torch.cat(
            [conditioned, herb, conditioned * herb], dim=-1)).squeeze(-1)
        mechanism = self.mechanism(pair_mechanism).squeeze(-1)

        table, available = memory_table
        probability = table[herb_index, drug_index].clamp(1e-4, 1 - 1e-4)
        memory = self.memory_scale * torch.logit(probability) + self.memory_bias
        # exactly as reference: with no support behind a cell the channel emits
        # nothing at all, rather than a learned bias
        support = (available[herb_index, drug_index] > EPS).float()
        if self.cfg.no_support_bias:
            memory = memory * support + self.no_support_value * (1.0 - support)
        else:
            memory = memory * support

        residual = ((conditioned * self.residual_weight[drug_index]).sum(-1)
                    + self.residual_bias[drug_index])
        cold = None
        cold_term = None
        if self.cfg.cold_expert:
            cold = self.cold_head(torch.cat(
                [self._raw_herb[herb_index],
                 self._raw_drug[drug_index]], dim=-1)).squeeze(-1)
            avail = available[herb_index, drug_index]
            nh = getattr(self, "_n_obs_herb", None)
                                    
            nd = getattr(self, "_n_obs_drug", None)
            row = (nh[herb_index] if nh is not None
                   else torch.zeros_like(avail))
            col = (nd[drug_index] if nd is not None
                   else torch.zeros_like(avail))
            feats = torch.stack([torch.log1p(avail.clamp_min(0.0)),
                                 torch.log1p(row.clamp_min(0.0)),
                                 torch.log1p(col.clamp_min(0.0)),
                                 (avail > EPS).float()], dim=-1)
            if not self.cfg.hard_cold_route:
                gate = torch.sigmoid(self.label_gate(feats).squeeze(-1))
                cold_term = (1.0 - gate) * cold
                self.last_gate = gate.detach()
            self.last_cold_logit = cold
        channels = torch.stack([structural, mechanism, memory], dim=-1)
        self.last_channels = channels.detach()
        if self.cfg.fusion in ("B", "C"):
            # a channel that is fifteen times wider than another cannot receive
            # a comparable gradient, so each is divided by its own running
            # spread; the statistic is detached and never carries gradient
            if self.training:
                spread = channels.detach().std(dim=0).clamp_min(1e-3)
                self.channel_balance.mul_(0.9).add_(0.1 * spread)
            channels = channels / self.channel_balance.clamp_min(1e-3)
        out = (channels * self.channel_scale).sum(dim=-1) + residual
        if self.cfg.cold_expert and self.cfg.hard_cold_route:
            direct_drug_support = self._n_obs_drug[drug_index]
            out = torch.where(direct_drug_support <= EPS, cold, out)
        elif cold_term is not None:
            out = out + cold_term
        self.last_base_logit = out
        graph_contribution = torch.zeros_like(out)
        if self.cfg.graph_expert:
            left = self._graph_herb[herb_index]
            right = self._graph_drug[drug_index]
            graph_logit = self.graph_pair(torch.cat(
                [left, right, left * right, torch.abs(left - right)],
                dim=-1)).squeeze(-1)
            direct_herb_support = self._n_obs_herb[herb_index]
            direct_drug_support = self._n_obs_drug[drug_index]
            route = ((direct_herb_support > EPS) &
                     (direct_drug_support > EPS)).to(graph_logit.dtype)
            graph_contribution = self.graph_scale * graph_logit * route
            out = out + graph_contribution
        self.last_graph_contribution = graph_contribution
        return out
