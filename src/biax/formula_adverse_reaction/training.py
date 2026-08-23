from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from .data import Bench, Cohort, Context
from .metrics import pooled_metrics
from .model import Config, BiAxADR

def _tensors(cohort: Cohort, bench: Bench, device):
    mat = torch.as_tensor(cohort.material_feature, device=device)
    idx = torch.as_tensor(cohort.material_index, device=device)
    mat_feat = mat[idx]
    dose = torch.as_tensor(cohort.material_dose, device=device)
    mask = torch.as_tensor(cohort.material_mask, device=device)
    mech = torch.as_tensor(cohort.pair_mech, device=device)
    presence = torch.as_tensor(cohort.presence, device=device)
    comp_dose = torch.as_tensor(cohort.dose, device=device)
    return (mat_feat, dose, mask, mech, presence, comp_dose)

class Runner:

    def __init__(self, bench: Bench, cfg: Config, device: str='cpu'):
        self.bench = bench
        self.cfg = cfg
        self.device = torch.device(device)
        self.main = _tensors(bench.main, bench, self.device)
        self.external = _tensors(bench.external, bench, self.device)
        self.adr = torch.as_tensor(bench.endpoints.feature, device=self.device)
        self.hier = torch.as_tensor(bench.endpoints.hierarchy, device=self.device)
        self.pf = torch.as_tensor(bench.main_pair_formula, device=self.device)
        self.pa = torch.as_tensor(bench.main_pair_adr, device=self.device)
        self.y = torch.as_tensor(bench.main_y.astype(np.float32), device=self.device)
        self.epf = torch.as_tensor(bench.external_pair_formula, device=self.device)
        self.epa = torch.as_tensor(bench.external_pair_adr, device=self.device)

    def _memory_matrices(self, fit_rows: np.ndarray):
        F_, A = (self.bench.main.n_formula, len(self.bench.endpoints.adr_ids))
        labels = torch.zeros(F_, A, device=self.device)
        observed = torch.zeros(F_, A, device=self.device)
        rows = torch.as_tensor(np.asarray(fit_rows), device=self.device)
        f = self.pf[rows]
        a = self.pa[rows]
        labels[f, a] = self.y[rows]
        observed[f, a] = 1.0
        return (labels, observed)

    def _new(self, seed: int) -> BiAxADR:
        torch.manual_seed(seed)
        np.random.seed(seed % 2 ** 31)
        model = BiAxADR(d_material=self.bench.main.material_feature.shape[1], d_endpoint=self.bench.endpoints.feature.shape[1], n_mech=self.bench.main.pair_mech.shape[-1], cfg=self.cfg, n_endpoint=len(self.bench.endpoints.adr_ids), n_material=self.bench.main.presence.shape[1]).to(self.device)
        return model

    def fit(self, train_rows: np.ndarray, seed: int,
            validation_rows: np.ndarray | None = None,
            max_epochs: int = 600, patience: int = 60,
            fixed_epochs: int | None = None):
        train_np = np.asarray(train_rows, dtype=np.int64)
        if train_np.size == 0:
            raise ValueError('empty training rows')
        fit_rows = torch.as_tensor(train_np, device=self.device)
        stop_rows = None
        if validation_rows is not None:
            validation_np = np.asarray(validation_rows, dtype=np.int64)
            if validation_np.size == 0:
                raise ValueError('empty validation rows')
            stop_rows = torch.as_tensor(validation_np, device=self.device)
        if fixed_epochs is None and stop_rows is None:
            raise ValueError('early stopping requires validation rows')
        total_epochs = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)
        if total_epochs < 1:
            raise ValueError('epochs must be positive')
        model = self._new(seed)
        opt = torch.optim.AdamW(model.param_groups(lr=0.003, weight_decay=0.01))
        if self.cfg.pos_weight < 0:
            pos = float(self.y[fit_rows].sum())
            neg = float(len(fit_rows) - pos)
            value = min(max(neg / max(pos, 1.0), 1.0), 20.0)
        else:
            value = float(self.cfg.pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(value, device=self.device))
        if self.cfg.early_stopping_metric not in ('loss', 'aupr'):
            raise ValueError('early_stopping_metric must be loss or aupr')
        best = float('inf') if self.cfg.early_stopping_metric == 'loss' else -float('inf')
        best_state, best_epoch, bad = (None, 0, 0)
        mat_feat, dose, mask, mech, presence, comp_dose = self.main
        labels, observed = self._memory_matrices(train_np)
        n_vf, n_va = (len(model.mat_views), len(model.adr_views))
        grid = [(i, j) for i in range(n_vf) for j in range(n_va)]
        vrng = np.random.default_rng(seed + 1)
        n_train_views = max(1, min(int(self.cfg.train_views), len(grid)))
        history = []
        for epoch in range(total_epochs):
            model.train()
            opt.zero_grad()
            picks = [grid[i] for i in vrng.choice(len(grid), n_train_views, replace=False)]
            if self.cfg.support_dropout > 0:
                keep = torch.as_tensor(vrng.random(observed.shape[0]) >= self.cfg.support_dropout, device=self.device).float().unsqueeze(-1)
                step_observed, step_labels = (observed * keep, labels * keep)
            else:
                step_observed, step_labels = (observed, labels)
            if self.cfg.endpoint_support_dropout > 0:
                endpoint_keep = torch.as_tensor(
                    vrng.random(observed.shape[1]) >= self.cfg.endpoint_support_dropout,
                    device=self.device,
                ).float().unsqueeze(0)
                step_observed = step_observed * endpoint_keep
                step_labels = step_labels * endpoint_keep
            per_view, gate, aux = ([], None, None)
            for v in picks:
                lg, _, g, a = model(
                    mat_feat, dose, mask, self.adr, mech, step_labels, step_observed,
                    hierarchy=self.hier, presence=presence, comp_dose=comp_dose,
                    views=v, balance_pairs=(self.pa[fit_rows], self.pf[fit_rows]),
                )
                per_view.append(lg[self.pf, self.pa])
                gate, aux = (g, a)
            stack = torch.stack(per_view)
            flat = model.aggregate_views(stack)
            loss = loss_fn(flat[fit_rows], self.y[fit_rows]) + model.penalty(gate)
            for pv in per_view:
                loss = loss + self.cfg.view_aux * loss_fn(pv[fit_rows], self.y[fit_rows])
            if self.cfg.aux_weight > 0:
                for branch in aux.values():
                    bflat = branch[self.pf, self.pa]
                    loss = loss + self.cfg.aux_weight * loss_fn(bflat[fit_rows], self.y[fit_rows])
            if self.cfg.struct_aux_weight > 0:
                structural = aux['struct'][self.pf, self.pa]
                loss = loss + self.cfg.struct_aux_weight * loss_fn(
                    structural[fit_rows], self.y[fit_rows]
                )
            if self.cfg.view_soft_label > 0 and len(per_view) > 1:
                target = torch.sigmoid(flat).detach()
                for pv in per_view:
                    loss = loss + self.cfg.view_soft_label * F.mse_loss(torch.sigmoid(pv[fit_rows]), target[fit_rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if fixed_epochs is not None:
                history.append({'epoch': epoch + 1, 'training_loss': float(loss.detach().cpu())})
                continue
            model.eval()
            with torch.no_grad():
                ev = []
                for vv in grid:
                    lg, _, _, _ = model(mat_feat, dose, mask, self.adr, mech, labels, observed, hierarchy=self.hier, presence=presence, comp_dose=comp_dose, views=vv)
                    ev.append(lg[self.pf, self.pa])
                fe = model.aggregate_views(torch.stack(ev))
                validation_logits = fe[stop_rows]
                validation_loss = float(loss_fn(validation_logits, self.y[stop_rows]))
                validation_probability = torch.sigmoid(validation_logits).detach().cpu().numpy()
                validation_label = self.y[stop_rows].detach().cpu().numpy()
                validation_aupr = float(average_precision_score(validation_label, validation_probability))
            history.append({'epoch': epoch + 1, 'training_loss': float(loss.detach().cpu()),
                            'validation_loss': validation_loss,
                            'validation_aupr': validation_aupr})
            score = validation_loss if self.cfg.early_stopping_metric == 'loss' else validation_aupr
            improved = (score < best - 1e-05) if self.cfg.early_stopping_metric == 'loss' else (score > best + 1e-05)
            if improved:
                best, bad, best_epoch = (score, 0, epoch + 1)
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if fixed_epochs is not None:
            best_epoch = total_epochs
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        model.fit_labels, model.fit_observed = (labels, observed)
        return model, {
            'best_epoch': int(best_epoch),
            'epochs_completed': len(history),
            'epoch_limit_reached': bool(fixed_epochs is None and len(history) >= max_epochs),
            'max_epochs': int(max_epochs),
            'patience': int(patience),
            'history': history,
        }

    def _view_grid(self, model):
        if not self.cfg.use_multiview:
            return [(-1, -1)]
        return [(i, j) for i in range(len(model.mat_views)) for j in range(len(model.adr_views))]

    @torch.no_grad()
    def predict_main(self, model: BiAxADR, rows: np.ndarray) -> np.ndarray:
        mat_feat, dose, mask, mech, presence, comp_dose = self.main
        idx = torch.as_tensor(rows, device=self.device)
        acc = []
        for v in self._view_grid(model):
            logits, _, _, _ = model(mat_feat, dose, mask, self.adr, mech, model.fit_labels, model.fit_observed, hierarchy=self.hier, presence=presence, comp_dose=comp_dose, views=v)
            acc.append(logits[self.pf, self.pa])
        agg = torch.sigmoid(model.aggregate_views(torch.stack(acc)))
        return agg[idx].cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def predict_external(self, model: BiAxADR, rows: np.ndarray) -> np.ndarray:
        mat_feat, dose, mask, mech, presence, comp_dose = self.external
        ref = self.main[:3]
        idx = torch.as_tensor(rows, device=self.device)
        acc = []
        for v in self._view_grid(model):
            logits, _, _, _ = model(mat_feat, dose, mask, self.adr, mech, model.fit_labels, model.fit_observed, ref=ref, hierarchy=self.hier, presence=presence, comp_dose=comp_dose, views=v)
            acc.append(logits[self.epf, self.epa])
        agg = torch.sigmoid(model.aggregate_views(torch.stack(acc)))
        return agg[idx].cpu().numpy().astype(np.float64)
