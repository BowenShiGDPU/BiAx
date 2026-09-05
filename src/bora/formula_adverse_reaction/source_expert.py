from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    return np.log(probability) - np.log1p(-probability)


def source_routed_probability(core_probability: np.ndarray, expert_logit: np.ndarray,
                              scale: float, source_gate: np.ndarray) -> np.ndarray:
    """Apply the source expert only where the source gate is active."""
    core = np.asarray(core_probability, dtype=np.float64)
    expert = np.asarray(expert_logit, dtype=np.float64)
    gate = np.asarray(source_gate, dtype=bool)
    output = core.copy()
    output[gate] = 1.0 / (1.0 + np.exp(-np.clip(
        probability_to_logit(core[gate]) + scale * expert[gate], -40, 40
    )))
    return output


class FormulaSourceExpert(nn.Module):
    def __init__(self, n_material: int, endpoint_dim: int, n_endpoint: int,
                 rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.formula = nn.Linear(n_material, rank, bias=False)
        self.endpoint_norm = nn.LayerNorm(endpoint_dim)
        self.endpoint = nn.Linear(endpoint_dim, rank, bias=False)
        self.formula_marginal = nn.Linear(n_material, 1, bias=False)
        self.endpoint_bias = nn.Parameter(torch.zeros(n_endpoint))
        self.dropout = nn.Dropout(dropout)
        self.rank = rank

    def forward(self, presence: torch.Tensor, endpoint_feature: torch.Tensor) -> torch.Tensor:
        formula_state = self.dropout(F.gelu(self.formula(presence)))
        endpoint_state = self.dropout(F.gelu(
            self.endpoint(self.endpoint_norm(endpoint_feature))
        ))
        interaction = formula_state @ endpoint_state.T / math.sqrt(self.rank)
        return interaction + self.formula_marginal(presence) + self.endpoint_bias.unsqueeze(0)
