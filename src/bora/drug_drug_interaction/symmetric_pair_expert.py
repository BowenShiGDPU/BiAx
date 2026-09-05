from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ExpertConfig:
    channel_width: int = 32
    drug_width: int = 64
    hidden_width: int = 64
    dropout: float = 0.10
    withhold_fraction: float = 0.20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 4096
    max_epochs: int = 80
    patience: int = 10
    scale_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)


class RawDrugEncoder(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], cfg: ExpertConfig) -> None:
        super().__init__()
        self.channels = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(size), nn.Linear(size, cfg.channel_width), nn.GELU())
            for size in dimensions
        ])
        self.fuse = nn.Sequential(
            nn.Linear(3 * cfg.channel_width, cfg.drug_width), nn.GELU(), nn.Dropout(cfg.dropout)
        )

    def forward(self, semantic: torch.Tensor, graph: torch.Tensor,
                structure: torch.Tensor) -> torch.Tensor:
        states = [module(value) for module, value in zip(self.channels, (semantic, graph, structure))]
        return self.fuse(torch.cat(states, dim=-1))


class SymmetricDrugPairExpert(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], cfg: ExpertConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ExpertConfig()
        self.drug = RawDrugEncoder(dimensions, self.cfg)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * self.cfg.drug_width),
            nn.Linear(3 * self.cfg.drug_width, self.cfg.hidden_width),
            nn.GELU(), nn.Dropout(self.cfg.dropout), nn.Linear(self.cfg.hidden_width, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode_all(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.drug(*features)

    def score_from_states(self, state: torch.Tensor, left: torch.Tensor,
                          right: torch.Tensor) -> torch.Tensor:
        first, second = state[left], state[right]
        pair = torch.cat([first + second, (first - second).abs(), first * second], dim=-1)
        return self.head(pair).squeeze(-1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.score_from_states(self.encode_all(features), left, right)


def route_mask(left: np.ndarray, right: np.ndarray, training_rows: np.ndarray,
               query_rows: np.ndarray, n_entity: int) -> np.ndarray:
    degree = np.bincount(np.r_[left[training_rows], right[training_rows]], minlength=n_entity)
    return (degree[left[query_rows]] == 0) | (degree[right[query_rows]] == 0)


def routed_probability(core_probability: np.ndarray, expert_logit: np.ndarray,
                       route: np.ndarray, scale: float) -> np.ndarray:
    core = np.asarray(core_probability, dtype=np.float64)
    output = core.copy()
    if scale == 0.0 or not np.any(route):
        return output
    clipped = np.clip(core[route], 1.0e-7, 1.0 - 1.0e-7)
    core_logit = np.log(clipped) - np.log1p(-clipped)
    output[route] = 1.0 / (1.0 + np.exp(-(core_logit + scale * expert_logit[route])))
    return output
