from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .data import TaskData
from .metrics import binary_metrics, choose_threshold, multiclass_metrics
from .model import BORADrugInteraction, ModelConfig


class Trainer:
    def __init__(self, data: TaskData, cfg: ModelConfig, device: str) -> None:
        self.data = data
        self.cfg = cfg
        self.device = torch.device(device)
        entity = data.entity
        self.entity_features = {
            "semantic": torch.as_tensor(entity.semantic, device=self.device),
            "graph": torch.as_tensor(entity.graph, device=self.device),
            "structure": torch.as_tensor(entity.structure, device=self.device),
        }
        self.endpoint_features = None
        if data.endpoint is not None:
            self.endpoint_features = {
                "semantic": torch.as_tensor(data.endpoint.semantic, device=self.device),
                "graph": torch.as_tensor(data.endpoint.graph, device=self.device),
                "mechanism": torch.as_tensor(data.endpoint.mechanism, device=self.device),
            }
        self.left = torch.as_tensor(data.left, dtype=torch.long, device=self.device)
        self.right = torch.as_tensor(data.right, dtype=torch.long, device=self.device)
        self.label = torch.as_tensor(data.label, dtype=torch.long, device=self.device)
        self.classes = np.unique(data.label)

    def new_model(self, seed: int) -> BORADrugInteraction:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.data.task == "single_adr":
            model = BORADrugInteraction(
                self.data.task, len(self.data.entity.ids), len(self.data.endpoint.ids), cfg=self.cfg
            )
        elif self.data.task == "ddi_type86":
            model = BORADrugInteraction(
                self.data.task, len(self.data.entity.ids), n_class=len(self.classes), cfg=self.cfg
            )
        else:
            model = BORADrugInteraction(self.data.task, len(self.data.entity.ids), cfg=self.cfg)
        return model.to(self.device)

    def memory(self, train_rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(train_rows, dtype=torch.long, device=self.device)
        left, right, label = self.left[rows], self.right[rows], self.label[rows]
        n = len(self.data.entity.ids)
        if self.data.task == "single_adr":
            m = len(self.data.endpoint.ids)
            labels = torch.zeros(n, m, device=self.device)
            observed = torch.zeros(n, m, device=self.device)
            labels[left, right] = label.float()
            observed[left, right] = 1.0
            return labels, observed
        if self.data.task == "ddi_binary":
            labels = torch.zeros(n, n, device=self.device)
            observed = torch.zeros(n, n, device=self.device)
            labels[left, right] = labels[right, left] = label.float()
            observed[left, right] = observed[right, left] = 1.0
            return labels, observed
        counts = torch.zeros(n, n, len(self.classes), device=self.device)
        class_index = label - int(self.classes.min())
        ones = torch.ones(len(rows), device=self.device)
        counts.index_put_((left, right, class_index), ones, accumulate=True)
        total = counts.sum(dim=-1)
        distribution = counts / total.unsqueeze(-1).clamp_min(1.0)
        return distribution, (total > 0).float()

    def _support_dropout(self, labels: torch.Tensor, observed: torch.Tensor,
                         generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.support_dropout <= 0:
            return labels, observed
        keep = (
            torch.rand(observed.shape[0], generator=generator, device=self.device)
            >= self.cfg.support_dropout
        ).float()
        if observed.ndim == 2:
            dropped_observed = observed * keep[:, None]
            if self.data.task != "single_adr":
                dropped_observed = dropped_observed * keep[None, :]
        else:
            raise ValueError("unexpected observed rank")
        return labels, dropped_observed

    def _mechanism(self, rows: np.ndarray) -> torch.Tensor:
        left = self.data.left[rows]
        right = self.data.right[rows]
        values = np.asarray(self.data.pair_mechanism[left, right], dtype=np.float32)
        return torch.as_tensor(values, device=self.device)

    def _logits(self, model: BORADrugInteraction, rows: np.ndarray, labels: torch.Tensor,
                observed: torch.Tensor) -> torch.Tensor:
        index = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        left, right = self.left[index], self.right[index]
        mechanism = self._mechanism(rows)
        entity_tokens, entity_state = model.encode_entities(self.entity_features)
        if self.data.task == "single_adr":
            _, endpoint_state = model.encode_endpoints(self.endpoint_features)
            return model.score_single_adr(
                entity_tokens, entity_state, endpoint_state, left, right,
                mechanism, labels, observed,
            )
        if self.data.task == "ddi_binary":
            return model.score_ddi_binary(
                entity_tokens, entity_state, left, right, mechanism, labels, observed
            )
        return model.score_ddi_type86(
            entity_tokens, entity_state, left, right, mechanism, labels, observed
        )

    def _epoch_rows(self, train_rows: np.ndarray, seed: int, epoch: int,
                    negative_ratio: int) -> np.ndarray:
        if self.data.task != "single_adr" or negative_ratio <= 0:
            result = np.asarray(train_rows).copy()
            np.random.default_rng(seed + epoch * 1009).shuffle(result)
            return result
        labels = self.data.label[train_rows]
        positive = np.asarray(train_rows)[labels == 1]
        negative = np.asarray(train_rows)[labels == 0]
        rng = np.random.default_rng(seed + epoch * 1009)
        count = min(len(negative), negative_ratio * len(positive))
        sampled = rng.choice(negative, size=count, replace=False)
        result = np.concatenate([positive, sampled])
        rng.shuffle(result)
        return result

    def binary_pos_weight(self, train_rows: np.ndarray, negative_ratio: int) -> float:
        """Match the capped full-data class objective after negative sampling.

        Uniformly sampling negatives changes their inclusion probability.  The
        positive weight is therefore reduced by the same factor so that
        sampling and class weighting do not correct the imbalance twice.
        """
        train_label = self.data.label[train_rows]
        positives = max(int(train_label.sum()), 1)
        negatives = max(len(train_label) - positives, 0)
        full_weight = min(20.0, negatives / positives)
        if (
            self.data.task == "single_adr"
            and negative_ratio > 0
            and negatives > negative_ratio * positives
        ):
            negative_inclusion = (negative_ratio * positives) / negatives
            return float(full_weight * negative_inclusion)
        return float(full_weight)

    @torch.inference_mode()
    def predict(self, model: BORADrugInteraction, rows: np.ndarray, labels: torch.Tensor,
                observed: torch.Tensor, batch_size: int) -> np.ndarray:
        model.eval()
        output = []
        for start in range(0, len(rows), batch_size):
            logits = self._logits(model, rows[start:start + batch_size], labels, observed)
            probability = torch.softmax(logits, dim=-1) if self.data.task == "ddi_type86" else torch.sigmoid(logits)
            output.append(probability.cpu().numpy())
        return np.concatenate(output, axis=0)

    def fit(self, train_rows: np.ndarray, validation_rows: np.ndarray, seed: int,
            max_epochs: int = 30, patience: int = 5, batch_size: int = 8192,
            negative_ratio: int = 4) -> tuple[BORADrugInteraction, torch.Tensor, torch.Tensor, list[dict]]:
        model = self.new_model(seed)
        labels, observed = self.memory(train_rows)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        if self.data.task == "ddi_type86":
            train_label = self.data.label[train_rows] - int(self.classes.min())
            counts = np.bincount(train_label, minlength=len(self.classes))
            weights = np.minimum(20.0, len(train_label) / (len(self.classes) * np.maximum(counts, 1)))
            loss_function = nn.CrossEntropyLoss(
                weight=torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            )
        else:
            loss_function = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [self.binary_pos_weight(train_rows, negative_ratio)], device=self.device
                )
            )
        best_score, best_state, stale = -np.inf, None, 0
        history: list[dict] = []
        support_generator = torch.Generator(device=self.device).manual_seed(seed + 17)
        for epoch in range(max_epochs):
            model.train()
            epoch_rows = self._epoch_rows(train_rows, seed, epoch, negative_ratio)
            running_loss, examples = 0.0, 0
            for start in range(0, len(epoch_rows), batch_size):
                rows = epoch_rows[start:start + batch_size]
                step_labels, step_observed = self._support_dropout(labels, observed, support_generator)
                logits = self._logits(model, rows, step_labels, step_observed)
                target = self.label[torch.as_tensor(rows, device=self.device)]
                if self.data.task == "ddi_type86":
                    target = target - int(self.classes.min())
                    loss = loss_function(logits, target)
                else:
                    loss = loss_function(logits, target.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running_loss += float(loss.detach()) * len(rows)
                examples += len(rows)
            probability = self.predict(model, validation_rows, labels, observed, batch_size)
            if self.data.task == "ddi_type86":
                score = multiclass_metrics(
                    self.data.label[validation_rows], probability, self.classes
                )["aupr"]
            else:
                score = float(average_precision_score(self.data.label[validation_rows], probability))
            history.append({"epoch": epoch + 1, "train_loss": running_loss / max(examples, 1), "validation_aupr": score})
            if score > best_score + 1e-5:
                best_score, stale = score, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if epoch + 1 >= 3 and stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("training produced no valid checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        return model, labels, observed, history

    def evaluate(self, model: BORADrugInteraction, labels: torch.Tensor, observed: torch.Tensor,
                 validation_rows: np.ndarray, test_rows: np.ndarray,
                 batch_size: int) -> tuple[
                     dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None
                 ]:
        validation_probability = self.predict(
            model, validation_rows, labels, observed, batch_size
        )
        test_probability = self.predict(model, test_rows, labels, observed, batch_size)
        if self.data.task == "ddi_type86":
            validation_prediction = self.classes[np.argmax(validation_probability, axis=1)]
            test_prediction = self.classes[np.argmax(test_probability, axis=1)]
            metrics = {
                "validation": multiclass_metrics(
                    self.data.label[validation_rows], validation_probability, self.classes
                ),
                "test": multiclass_metrics(
                    self.data.label[test_rows], test_probability, self.classes
                ),
            }
            return (
                metrics, validation_probability, validation_prediction,
                test_probability, test_prediction, None,
            )
        threshold = choose_threshold(self.data.label[validation_rows], validation_probability)
        validation_prediction = (validation_probability >= threshold).astype(np.int8)
        test_prediction = (test_probability >= threshold).astype(np.int8)
        metrics = {
            "validation": binary_metrics(
                self.data.label[validation_rows], validation_probability, threshold
            ),
            "test": binary_metrics(self.data.label[test_rows], test_probability, threshold),
        }
        return (
            metrics, validation_probability, validation_prediction,
            test_probability, test_prediction, threshold,
        )


def config_dict(cfg: ModelConfig) -> dict:
    return asdict(cfg)
