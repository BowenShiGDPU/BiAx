from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .retrieval import ObservedRelationRetrieval


@dataclass(frozen=True)
class ModelConfig:
    semantic_dim: int = 1024
    graph_dim: int = 256
    structure_dim: int = 2049
    mechanism_dim: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    memory_topk_left: int = 20
    memory_topk_right: int = 10
    memory_tau_init: float = -1.0
    support_dropout: float = 0.3


class MultiChannelEntityEncoder(nn.Module):
    """Encode semantic, graph-relation and molecular-structure channels as tokens."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        dimensions = (cfg.semantic_dim, cfg.graph_dim, cfg.structure_dim)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(size), nn.Linear(size, d), nn.GELU(), nn.Dropout(cfg.dropout))
            for size in dimensions
        ])
        self.channel_embedding = nn.Parameter(torch.zeros(len(dimensions), d))
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

    def forward(self, semantic: torch.Tensor, graph: torch.Tensor,
                structure: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([
            projection(value)
            for projection, value in zip(self.projections, (semantic, graph, structure))
        ], dim=1)
        tokens = self.blocks(tokens + self.channel_embedding.unsqueeze(0))
        tokens = self.norm(tokens)
        return tokens, tokens.mean(dim=1)


class MultiChannelEndpointEncoder(nn.Module):
    """Encode endpoint identity, graph relation and mechanism text channels."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        dimensions = (cfg.semantic_dim, cfg.graph_dim, cfg.semantic_dim)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(size), nn.Linear(size, d), nn.GELU(), nn.Dropout(cfg.dropout))
            for size in dimensions
        ])
        self.channel_embedding = nn.Parameter(torch.zeros(len(dimensions), d))
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

    def forward(self, semantic: torch.Tensor, graph: torch.Tensor,
                mechanism: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([
            projection(value)
            for projection, value in zip(self.projections, (semantic, graph, mechanism))
        ], dim=1)
        tokens = self.blocks(tokens + self.channel_embedding.unsqueeze(0))
        tokens = self.norm(tokens)
        return tokens, tokens.mean(dim=1)


class ConditionalReadout(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, condition: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        value, _ = self.attention(condition.unsqueeze(1), tokens, tokens, need_weights=False)
        return self.norm(condition + value.squeeze(1))


class SymmetricPairEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.readout = ConditionalReadout(cfg)
        self.mix = nn.Sequential(
            nn.LayerNorm(3 * cfg.d_model), nn.Linear(3 * cfg.d_model, cfg.d_model), nn.GELU()
        )

    def forward(self, a_tokens: torch.Tensor, a_state: torch.Tensor,
                b_tokens: torch.Tensor, b_state: torch.Tensor) -> torch.Tensor:
        a_given_b = self.readout(b_state, a_tokens)
        b_given_a = self.readout(a_state, b_tokens)
        return self.mix(torch.cat([
            a_given_b + b_given_a,
            a_given_b * b_given_a,
            (a_given_b - b_given_a).abs(),
        ], dim=-1))


class DirectedPairEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.readout = ConditionalReadout(cfg)
        self.left_role = nn.Linear(cfg.d_model, cfg.d_model)
        self.right_role = nn.Linear(cfg.d_model, cfg.d_model)
        self.mix = nn.Sequential(
            nn.LayerNorm(4 * cfg.d_model), nn.Linear(4 * cfg.d_model, cfg.d_model), nn.GELU()
        )

    def forward(self, left_tokens: torch.Tensor, left_state: torch.Tensor,
                right_tokens: torch.Tensor, right_state: torch.Tensor) -> torch.Tensor:
        left = self.left_role(self.readout(right_state, left_tokens))
        right = self.right_role(self.readout(left_state, right_tokens))
        return self.mix(torch.cat([left, right, left * right, left - right], dim=-1))


class BORADrugInteraction(nn.Module):
    """BORA drug encoder and prediction heads for binary and endpoint labels."""

    VALID_TASKS = {"single_adr", "ddi_binary", "ddi_type86"}

    def __init__(self, task: str, n_entity: int, n_endpoint: int = 0,
                 n_class: int = 0, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        if task not in self.VALID_TASKS:
            raise ValueError(f"unsupported task: {task}")
        if task == "single_adr" and n_endpoint <= 0:
            raise ValueError("single_adr requires endpoints")
        if task == "ddi_type86" and n_class <= 1:
            raise ValueError("ddi_type86 requires multiple classes")
        self.task = task
        self.cfg = cfg or ModelConfig()
        d = self.cfg.d_model
        self.entity_encoder = MultiChannelEntityEncoder(self.cfg)
        self.endpoint_encoder = MultiChannelEndpointEncoder(self.cfg) if task == "single_adr" else None
        self.single_readout = ConditionalReadout(self.cfg) if task == "single_adr" else None
        self.symmetric_pair = SymmetricPairEncoder(self.cfg) if task == "ddi_binary" else None
        self.directed_pair = DirectedPairEncoder(self.cfg) if task == "ddi_type86" else None
        self.memory = ObservedRelationRetrieval(
            d, self.cfg.memory_topk_left, self.cfg.memory_topk_right,
            self.cfg.memory_tau_init,
        )
        self.structural = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(),
            nn.Dropout(self.cfg.dropout), nn.Linear(d, 1 if task != "ddi_type86" else n_class),
        ) if task == "single_adr" else nn.Linear(d, 1 if task == "ddi_binary" else n_class)
        output_dim = 1 if task != "ddi_type86" else n_class
        self.mechanism = nn.Sequential(
            nn.LayerNorm(self.cfg.mechanism_dim), nn.Linear(self.cfg.mechanism_dim, 16),
            nn.GELU(), nn.Linear(16, output_dim),
        )
        self.memory_scale = nn.Parameter(torch.ones(output_dim))
        self.memory_bias = nn.Parameter(torch.zeros(output_dim))
        self.channel_scale = nn.Parameter(torch.ones(3, output_dim))
        if task == "single_adr":
            self.residual_weight = nn.Parameter(torch.zeros(n_endpoint, d))
            self.residual_bias = nn.Parameter(torch.zeros(n_endpoint))
        elif task == "ddi_binary":
            self.residual_weight = nn.Parameter(torch.zeros(n_entity, d))
            self.residual_bias = nn.Parameter(torch.zeros(n_entity))
        else:
            self.residual_weight = nn.Parameter(torch.zeros(n_class, d))
            self.residual_bias = nn.Parameter(torch.zeros(n_class))

    def encode_entities(self, features: dict[str, torch.Tensor]):
        return self.entity_encoder(
            features["semantic"], features["graph"], features["structure"]
        )

    def encode_endpoints(self, features: dict[str, torch.Tensor]):
        if self.endpoint_encoder is None:
            raise RuntimeError("this task has no endpoint encoder")
        return self.endpoint_encoder(
            features["semantic"], features["graph"], features["mechanism"]
        )

    def _combine(self, structural: torch.Tensor, mechanism: torch.Tensor,
                 memory: torch.Tensor, residual: torch.Tensor,
                 support: torch.Tensor) -> torch.Tensor:
        if structural.ndim == 1:
            structural = structural.unsqueeze(-1)
            mechanism = mechanism.unsqueeze(-1) if mechanism.ndim == 1 else mechanism
            memory = memory.unsqueeze(-1) if memory.ndim == 1 else memory
            residual = residual.unsqueeze(-1) if residual.ndim == 1 else residual
        memory = memory * (support > 0).unsqueeze(-1)
        channels = torch.stack([structural, mechanism, memory], dim=1)
        logit = (0.5 * self.channel_scale.unsqueeze(0) * channels).sum(dim=1) + residual
        return logit.squeeze(-1) if logit.shape[-1] == 1 else logit

    def score_single_adr(self, entity_tokens: torch.Tensor, entity_state: torch.Tensor,
                         endpoint_state: torch.Tensor, pair_left: torch.Tensor,
                         pair_right: torch.Tensor, mechanism: torch.Tensor,
                         labels: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        readout = self.single_readout(
            endpoint_state[pair_right], entity_tokens[pair_left]
        )
        endpoint = endpoint_state[pair_right]
        structural = self.structural(torch.cat([readout, endpoint, readout * endpoint], dim=-1)).squeeze(-1)
        mech = self.mechanism(mechanism).squeeze(-1)
        probability, support = self.memory.rectangular_binary(
            entity_state[pair_left], endpoint_state[pair_right], entity_state,
            endpoint_state, pair_left, pair_right, labels, observed, True,
        )
        memory = self.memory_scale[0] * torch.logit(probability) + self.memory_bias[0]
        residual = (
            readout * self.residual_weight[pair_right]
        ).sum(dim=-1) + self.residual_bias[pair_right]
        return self._combine(structural, mech, memory, residual, support)

    def score_ddi_binary(self, entity_tokens: torch.Tensor, entity_state: torch.Tensor,
                         pair_left: torch.Tensor, pair_right: torch.Tensor,
                         mechanism: torch.Tensor, labels: torch.Tensor,
                         observed: torch.Tensor) -> torch.Tensor:
        pair = self.symmetric_pair(
            entity_tokens[pair_left], entity_state[pair_left],
            entity_tokens[pair_right], entity_state[pair_right],
        )
        structural = self.structural(pair).squeeze(-1)
        mech = self.mechanism(mechanism).squeeze(-1)
        probability, support = self.memory.symmetric_binary(
            entity_state, pair_left, pair_right, labels, observed
        )
        memory = self.memory_scale[0] * torch.logit(probability) + self.memory_bias[0]
        residual_weight = 0.5 * (
            self.residual_weight[pair_left] + self.residual_weight[pair_right]
        )
        residual = (pair * residual_weight).sum(dim=-1) + 0.5 * (
            self.residual_bias[pair_left] + self.residual_bias[pair_right]
        )
        return self._combine(structural, mech, memory, residual, support)

    def score_ddi_type86(self, entity_tokens: torch.Tensor, entity_state: torch.Tensor,
                         pair_left: torch.Tensor, pair_right: torch.Tensor,
                         mechanism: torch.Tensor, class_counts: torch.Tensor,
                         observed: torch.Tensor) -> torch.Tensor:
        pair = self.directed_pair(
            entity_tokens[pair_left], entity_state[pair_left],
            entity_tokens[pair_right], entity_state[pair_right],
        )
        structural = self.structural(pair)
        mech = self.mechanism(mechanism)
        probability, support = self.memory.directed_multiclass(
            entity_state, pair_left, pair_right, class_counts, observed, False
        )
        memory = self.memory_scale * probability.log() + self.memory_bias
        residual = pair @ self.residual_weight.t() + self.residual_bias
        return self._combine(structural, mech, memory, residual, support)
