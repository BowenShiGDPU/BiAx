from __future__ import annotations

import numpy as np
import torch


class UnseenHerbExpert(torch.nn.Module):
    def __init__(self, raw_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * raw_dim, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, herb: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([herb, drug], dim=-1)).squeeze(-1)


def route_mask(herb_index: np.ndarray, training_rows: np.ndarray,
               query_rows: np.ndarray, n_herb: int) -> np.ndarray:
    support = np.bincount(herb_index[training_rows], minlength=n_herb)
    return support[herb_index[query_rows]] == 0


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def routed_probability(core_probability: np.ndarray, expert_probability: np.ndarray,
                       route: np.ndarray, scale: float) -> np.ndarray:
    mixed = 1.0 / (1.0 + np.exp(-np.clip(
        (1 - scale) * probability_to_logit(core_probability)
        + scale * probability_to_logit(expert_probability), -40, 40
    )))
    return np.where(route, mixed, np.asarray(core_probability, dtype=np.float64))


def routed_probability_torch(core_probability: torch.Tensor,
                             expert_probability: torch.Tensor,
                             route: torch.Tensor,
                             scale: float) -> torch.Tensor:
    """Compose core and expert probabilities on routed cells.

    Both probabilities are clipped before conversion to logits, matching the
    numerical stability guard used by the evaluation pipeline. Non-routed
    cells are copied from the core prediction exactly.
    """
    core = core_probability.to(torch.float64).clamp(1e-7, 1 - 1e-7)
    expert = expert_probability.to(torch.float64).clamp(1e-7, 1 - 1e-7)
    mixed = torch.sigmoid(
        (1.0 - scale) * torch.logit(core) + scale * torch.logit(expert))
    return torch.where(route, mixed, core_probability.to(torch.float64))
