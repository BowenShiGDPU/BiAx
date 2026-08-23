from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BiAxisMemory(nn.Module):
    """Learned two-axis label memory with task-defined exact exclusions."""

    def __init__(self, d_model: int, topk_left: int = 20, topk_right: int = 10,
                 tau_init: float = -1.0) -> None:
        super().__init__()
        self.left = nn.Linear(d_model, d_model)
        self.right = nn.Linear(d_model, d_model)
        self.log_tau_left = nn.Parameter(torch.tensor(float(tau_init)))
        self.log_tau_right = nn.Parameter(torch.tensor(float(tau_init)))
        self.topk_left = int(topk_left)
        self.topk_right = int(topk_right)

    @staticmethod
    def _neighbors(query: torch.Tensor, reference: torch.Tensor, projection: nn.Linear,
                   log_tau: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(projection(query), dim=-1)
        reference = F.normalize(projection(reference), dim=-1)
        score = query @ reference.t() / log_tau.exp().clamp(0.02, 10.0)
        k = min(max(1, topk), reference.shape[0])
        value, index = score.topk(k, dim=-1)
        return index, torch.softmax(value, dim=-1)

    @staticmethod
    def _exclusion_mask(left_index: torch.Tensor, right_index: torch.Tensor,
                        query_left: torch.Tensor, query_right: torch.Tensor,
                        exclude_mirror: bool) -> torch.Tensor:
        li = left_index.unsqueeze(2)
        ri = right_index.unsqueeze(1)
        ql = query_left[:, None, None]
        qr = query_right[:, None, None]
        mask = (li == ql) & (ri == qr)
        if exclude_mirror:
            mask = mask | ((li == qr) & (ri == ql))
        return mask

    def rectangular_binary(
        self,
        query_left_state: torch.Tensor,
        query_right_state: torch.Tensor,
        reference_left_state: torch.Tensor,
        reference_right_state: torch.Tensor,
        query_left_index: torch.Tensor,
        query_right_index: torch.Tensor,
        labels: torch.Tensor,
        observed: torch.Tensor,
        exclude_self: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_index, left_weight = self._neighbors(
            query_left_state, reference_left_state, self.left,
            self.log_tau_left, self.topk_left,
        )
        right_index, right_weight = self._neighbors(
            query_right_state, reference_right_state, self.right,
            self.log_tau_right, self.topk_right,
        )
        local_observed = observed[left_index.unsqueeze(2), right_index.unsqueeze(1)]
        local_labels = labels[left_index.unsqueeze(2), right_index.unsqueeze(1)]
        weight = left_weight.unsqueeze(2) * right_weight.unsqueeze(1)
        if exclude_self:
            weight = weight.masked_fill(
                self._exclusion_mask(
                    left_index, right_index, query_left_index, query_right_index, False
                ),
                0.0,
            )
        effective = weight * local_observed
        denominator = effective.sum(dim=(1, 2))
        probability = (effective * local_labels).sum(dim=(1, 2)) / denominator.clamp_min(1e-9)
        return probability.clamp(1e-4, 1.0 - 1e-4), denominator

    def symmetric_binary(
        self,
        entity_state: torch.Tensor,
        query_left: torch.Tensor,
        query_right: torch.Tensor,
        labels: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def orientation(left_query: torch.Tensor, right_query: torch.Tensor):
            left_index, left_weight = self._neighbors(
                entity_state[left_query], entity_state, self.left,
                self.log_tau_left, self.topk_left,
            )
            right_index, right_weight = self._neighbors(
                entity_state[right_query], entity_state, self.right,
                self.log_tau_right, self.topk_right,
            )
            local_observed = observed[left_index.unsqueeze(2), right_index.unsqueeze(1)]
            local_labels = labels[left_index.unsqueeze(2), right_index.unsqueeze(1)]
            weight = left_weight.unsqueeze(2) * right_weight.unsqueeze(1)
            exclusion = self._exclusion_mask(
                left_index, right_index, query_left, query_right, True
            )
            effective = weight.masked_fill(exclusion, 0.0) * local_observed
            denominator = effective.sum(dim=(1, 2))
            numerator = (effective * local_labels).sum(dim=(1, 2))
            return numerator, denominator

        numerator_ab, denominator_ab = orientation(query_left, query_right)
        numerator_ba, denominator_ba = orientation(query_right, query_left)
        # Pool evidence from both orientations.  Averaging two conditional
        # probabilities would incorrectly halve the estimate when only one
        # orientation has support.
        denominator = denominator_ab + denominator_ba
        probability = (numerator_ab + numerator_ba) / denominator.clamp_min(1e-9)
        return probability.clamp(1e-4, 1.0 - 1e-4), denominator

    def directed_multiclass(
        self,
        entity_state: torch.Tensor,
        query_left: torch.Tensor,
        query_right: torch.Tensor,
        class_counts: torch.Tensor,
        observed: torch.Tensor,
        exclude_mirror: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_index, left_weight = self._neighbors(
            entity_state[query_left], entity_state, self.left,
            self.log_tau_left, self.topk_left,
        )
        right_index, right_weight = self._neighbors(
            entity_state[query_right], entity_state, self.right,
            self.log_tau_right, self.topk_right,
        )
        local_observed = observed[left_index.unsqueeze(2), right_index.unsqueeze(1)]
        local_counts = class_counts[left_index.unsqueeze(2), right_index.unsqueeze(1), :]
        weight = left_weight.unsqueeze(2) * right_weight.unsqueeze(1)
        exclusion = self._exclusion_mask(
            left_index, right_index, query_left, query_right, exclude_mirror
        )
        effective = weight.masked_fill(exclusion, 0.0) * local_observed
        class_evidence = (effective.unsqueeze(-1) * local_counts).sum(dim=(1, 2))
        denominator = class_evidence.sum(dim=-1)
        probability = class_evidence / denominator.unsqueeze(-1).clamp_min(1e-9)
        probability = probability.clamp_min(1e-6)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return probability, denominator
