from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    semantic_dim: int = 1024
    graph_dim: int = 256
    structure_dim: int = 2049
    endpoint_mechanism_dim: int = 1024
    pair_mechanism_dim: int = 9
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    memory_smoothing: float = 5.0
    memory_topk_drug: int = 20
    memory_topk_endpoint: int = 10
    memory_tau_init: float = -1.0
    support_dropout: float = 0.3
    support_warmup_epochs: int = 5
    retrieval_mix_init: float = -2.1972245773362196  # sigmoid = 0.10


class ChannelEncoder(nn.Module):
    """Represent three label-free evidence channels as an entity token set."""

    def __init__(self, dimensions: tuple[int, int, int], cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(size), nn.Linear(size, d), nn.GELU(), nn.Dropout(cfg.dropout)
            )
            for size in dimensions
        ])
        self.channel_embedding = nn.Parameter(torch.zeros(3, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=2 * d,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(d)

    def forward(
        self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([
            projection(value)
            for projection, value in zip(self.projections, (first, second, third))
        ], dim=1)
        tokens = self.norm(self.blocks(tokens + self.channel_embedding.unsqueeze(0)))
        return tokens, tokens.mean(dim=1)


class ConditionalReadout(nn.Module):
    """Read drug evidence conditionally on the queried ADR endpoint."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, condition: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        value, _ = self.attention(
            condition.unsqueeze(1), tokens, tokens, need_weights=False
        )
        return self.norm(condition + value.squeeze(1))


class BORAPairADR(nn.Module):
    """BORA drug-pair–ADR model with support-aware observed-relation retrieval."""

    def __init__(self, n_endpoint: int, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        d = self.cfg.d_model
        self.drug_encoder = ChannelEncoder(
            (self.cfg.semantic_dim, self.cfg.graph_dim, self.cfg.structure_dim), self.cfg
        )
        self.endpoint_encoder = ChannelEncoder(
            (self.cfg.semantic_dim, self.cfg.graph_dim, self.cfg.endpoint_mechanism_dim),
            self.cfg,
        )
        self.endpoint_readout = ConditionalReadout(self.cfg)
        self.pair_mix = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU()
        )
        self.structural = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(d, 1),
        )
        self.mechanism = nn.Sequential(
            nn.LayerNorm(self.cfg.pair_mechanism_dim),
            nn.Linear(self.cfg.pair_mechanism_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.drug_memory_projection = nn.Linear(d, d, bias=False)
        self.endpoint_memory_projection = nn.Linear(d, d, bias=False)
        self.log_tau_drug = nn.Parameter(torch.tensor(self.cfg.memory_tau_init))
        self.log_tau_endpoint = nn.Parameter(torch.tensor(self.cfg.memory_tau_init))
        self.retrieval_mix_logit = nn.Parameter(torch.tensor(self.cfg.retrieval_mix_init))
        self.memory_scale = nn.Parameter(torch.ones(()))
        self.memory_bias = nn.Parameter(torch.zeros(()))

        # Preserve the validated marginal anchor under the shared 0.5 fusion.
        # Structural and mechanism pathways remain trainable and acquire nonzero
        # scales from the first optimizer step without injecting random logits at
        # initialization.
        self.channel_scale = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))
        self.residual_weight = nn.Parameter(torch.zeros(n_endpoint, d))
        self.residual_bias = nn.Parameter(torch.zeros(n_endpoint))

    def encode_drugs(self, features: dict[str, torch.Tensor]):
        return self.drug_encoder(
            features["semantic"], features["graph"], features["structure"]
        )

    def encode_endpoints(self, features: dict[str, torch.Tensor]):
        return self.endpoint_encoder(
            features["semantic"], features["graph"], features["mechanism"]
        )

    @staticmethod
    def _sparse_weights(
        query: torch.Tensor,
        reference: torch.Tensor,
        projection: nn.Linear,
        log_tau: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(projection(query), dim=-1)
        r = F.normalize(projection(reference), dim=-1)
        score = q @ r.t() / log_tau.exp().clamp(0.02, 10.0)
        k = min(max(1, int(topk)), score.shape[-1])
        cut = score.topk(k, dim=-1).values[..., -1:]
        # Retain all ties at the sparse-softmax cutoff.
        active_count = (score >= cut).sum(dim=-1)
        max_active = int(active_count.max().item())
        similarity, index = score.topk(max_active, dim=-1)
        active = similarity >= cut
        weight = torch.softmax(
            similarity.masked_fill(~active, float("-inf")), dim=-1
        )
        return index, weight

    def neighbor_tables(
        self, drug_state: torch.Tensor, endpoint_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_index, drug_weight = self._sparse_weights(
            drug_state,
            drug_state,
            self.drug_memory_projection,
            self.log_tau_drug,
            self.cfg.memory_topk_drug,
        )
        endpoint_index, endpoint_weight = self._sparse_weights(
            endpoint_state,
            endpoint_state,
            self.endpoint_memory_projection,
            self.log_tau_endpoint,
            self.cfg.memory_topk_endpoint,
        )
        return drug_index, drug_weight, endpoint_index, endpoint_weight

    @staticmethod
    def _query_presence(
        drug: torch.Tensor,
        endpoint: torch.Tensor,
        row_keep: torch.Tensor | None,
        col_keep: torch.Tensor | None,
    ) -> torch.Tensor:
        present = torch.ones_like(drug, dtype=torch.float32)
        if row_keep is not None:
            present = present * row_keep[drug]
        if col_keep is not None:
            present = present * col_keep[endpoint]
        return present

    def _direct_probability(
        self,
        query_drug: torch.Tensor,
        endpoint: torch.Tensor,
        memory_positive: torch.Tensor,
        memory_observed: torch.Tensor,
        endpoint_prior: torch.Tensor,
        query_label: torch.Tensor | None,
        exclude_query: bool,
        row_keep: torch.Tensor | None,
        col_keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive = memory_positive[query_drug, endpoint]
        observed = memory_observed[query_drug, endpoint]
        if exclude_query:
            if query_label is None:
                raise ValueError("query_label is required for exact query exclusion")
            present = self._query_presence(query_drug, endpoint, row_keep, col_keep)
            positive = positive - present * query_label.float()
            observed = observed - present
        positive = positive.clamp_min(0.0)
        observed = observed.clamp_min(0.0)
        smoothing = self.cfg.memory_smoothing
        probability = (positive + smoothing * endpoint_prior[endpoint]) / (
            observed + smoothing
        ).clamp_min(1e-6)
        return probability, observed

    def _retrieval_probability(
        self,
        query_drug: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        endpoint: torch.Tensor,
        memory_positive: torch.Tensor,
        memory_observed: torch.Tensor,
        endpoint_prior: torch.Tensor,
        neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        query_label: torch.Tensor | None,
        exclude_query: bool,
        row_keep: torch.Tensor | None,
        col_keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drug_index, drug_weight, endpoint_index, endpoint_weight = neighbors
        local_drug = drug_index[query_drug]
        local_endpoint = endpoint_index[endpoint]
        positive = memory_positive[
            local_drug.unsqueeze(2), local_endpoint.unsqueeze(1)
        ]
        observed = memory_observed[
            local_drug.unsqueeze(2), local_endpoint.unsqueeze(1)
        ]
        if exclude_query:
            if query_label is None:
                raise ValueError("query_label is required for exact query exclusion")
            query_cell = (
                (
                    (local_drug == left.unsqueeze(1))
                    | (local_drug == right.unsqueeze(1))
                ).unsqueeze(2)
                & (local_endpoint == endpoint.unsqueeze(1)).unsqueeze(1)
            )
            present = torch.ones_like(observed)
            if row_keep is not None:
                present = present * row_keep[local_drug].unsqueeze(2)
            if col_keep is not None:
                present = present * col_keep[local_endpoint].unsqueeze(1)
            remove = query_cell.float() * present
            positive = positive - remove * query_label.float()[:, None, None]
            observed = observed - remove
        positive = positive.clamp_min(0.0)
        observed = observed.clamp_min(0.0)
        weight = (
            drug_weight[query_drug].unsqueeze(2)
            * endpoint_weight[endpoint].unsqueeze(1)
        )
        numerator = (weight * positive).sum(dim=(1, 2))
        support = (weight * observed).sum(dim=(1, 2)).clamp_min(0.0)
        smoothing = self.cfg.memory_smoothing
        probability = (numerator + smoothing * endpoint_prior[endpoint]) / (
            support + smoothing
        ).clamp_min(1e-6)
        return probability, support

    def _memory_for_drug(
        self,
        query_drug: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        endpoint: torch.Tensor,
        memory_positive: torch.Tensor,
        memory_observed: torch.Tensor,
        endpoint_prior: torch.Tensor,
        neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        query_label: torch.Tensor | None,
        exclude_query: bool,
        row_keep: torch.Tensor | None,
        col_keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direct, direct_support = self._direct_probability(
            query_drug,
            endpoint,
            memory_positive,
            memory_observed,
            endpoint_prior,
            query_label,
            exclude_query,
            row_keep,
            col_keep,
        )
        retrieved, retrieval_support = self._retrieval_probability(
            query_drug,
            left,
            right,
            endpoint,
            memory_positive,
            memory_observed,
            endpoint_prior,
            neighbors,
            query_label,
            exclude_query,
            row_keep,
            col_keep,
        )
        alpha = torch.sigmoid(self.retrieval_mix_logit)
        supported_mix = (1.0 - alpha) * direct + alpha * retrieved
        # Cold entities/endpoints have no direct cell: transfer becomes the full
        # memory pathway instead of an unconstrained signed correction.
        probability = torch.where(direct_support > 0, supported_mix, retrieved)
        support = direct_support + retrieval_support
        return probability, support

    def score_encoded(
        self,
        drug_tokens: torch.Tensor,
        drug_state: torch.Tensor,
        endpoint_tokens: torch.Tensor,
        endpoint_state: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        endpoint: torch.Tensor,
        pair_mechanism: torch.Tensor,
        memory_positive: torch.Tensor,
        memory_observed: torch.Tensor,
        endpoint_prior: torch.Tensor,
        neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        query_label: torch.Tensor | None = None,
        exclude_query: bool = False,
        row_keep: torch.Tensor | None = None,
        col_keep: torch.Tensor | None = None,
        residual_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        endpoint_condition = endpoint_state[endpoint]
        left_state = self.endpoint_readout(endpoint_condition, drug_tokens[left])
        right_state = self.endpoint_readout(endpoint_condition, drug_tokens[right])
        pair_state = self.pair_mix(torch.cat([
            left_state + right_state,
            left_state * right_state,
            (left_state - right_state).abs(),
        ], dim=-1))
        structural = self.structural(torch.cat([
            pair_state, endpoint_condition, pair_state * endpoint_condition
        ], dim=-1)).squeeze(-1)
        mechanism = self.mechanism(pair_mechanism).squeeze(-1)

        left_probability, left_support = self._memory_for_drug(
            left, left, right, endpoint, memory_positive, memory_observed,
            endpoint_prior, neighbors, query_label, exclude_query, row_keep, col_keep,
        )
        right_probability, right_support = self._memory_for_drug(
            right, left, right, endpoint, memory_positive, memory_observed,
            endpoint_prior, neighbors, query_label, exclude_query, row_keep, col_keep,
        )
        memory_probability = 0.5 * (left_probability + right_probability)
        memory_logit = torch.logit(memory_probability.clamp(1e-4, 1.0 - 1e-4))
        memory = self.memory_scale * memory_logit + self.memory_bias
        memory = memory * ((left_support + right_support) > 0).float()

        residual = (
            pair_state * self.residual_weight[endpoint]
        ).sum(dim=-1) + self.residual_bias[endpoint]
        if residual_keep is not None:
            residual = residual * residual_keep[endpoint]
        channels = torch.stack([structural, mechanism, memory], dim=-1)
        return (0.5 * channels * self.channel_scale).sum(dim=-1) + residual
