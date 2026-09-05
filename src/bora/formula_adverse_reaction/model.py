from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
EPS = 1e-08

@dataclass
class Config:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    ratio_bins: int = 16
    ratio_span: float = 3.0
    gate_l1: float = 0.001
    use_dsa: bool = False
    use_dose_gate: bool = False
    use_mech: bool = True
    use_material_mixing: bool = True
    use_readout: bool = True
    use_struct: bool = True
    use_adr_bias: bool = True
    use_support_fusion: bool = False
    mat_keep: tuple = ()
    adr_keep: tuple = ()
    use_memory: bool = True
    mem_topk_f: int = 20
    mem_topk_a: int = 10
    mem_tau_init: float = -1.0
    pos_weight: float = 1.0
    use_adr_residual: bool = True
    use_label_graph: bool = False
    label_graph_layers: int = 1
    use_composition: bool = False
    use_composition_sparse: bool = True
    comp_rank: int = 24
    comp_l1: float = 0.001
    comp_l2: float = 0.001
    comp_lr: float = 0.01
    aux_weight: float = 0.0
    support_dropout: float = 0.0
    endpoint_support_dropout: float = 0.3
    residual_support_gate: bool = True
    use_shared_bilinear: bool = False
    struct_aux_weight: float = 0.0
    residual_on_composition: bool = False
    use_channel_balance: bool = False
    channel_balance_momentum: float = 0.9
    use_multiview: bool = False
    view_consistency: float = 0.1
    view_topk: int = 4
    view_aux: float = 0.01
    view_soft_label: float = 0.5
    train_views: int = 3
    early_stopping_metric: str = 'aupr'

class MaterialSelfAttention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, h = (cfg.d_model, cfg.n_heads)
        assert d % h == 0
        self.h = h
        self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.ratio_bias = nn.Parameter(torch.zeros(cfg.ratio_bins + 1, h))
        edges = torch.linspace(-cfg.ratio_span, cfg.ratio_span, cfg.ratio_bins)
        self.register_buffer('ratio_edges', edges)

    def bias(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r = log_p.unsqueeze(-1) - log_p.unsqueeze(-2)
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        idx = torch.bucketize(r, self.ratio_edges)
        b = self.ratio_bias[idx]
        return b.permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.h, self.dh).transpose(1, 2)
        k = k.view(B, S, self.h, self.dh).transpose(1, 2)
        v = v.view(B, S, self.h, self.dh).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        if self.cfg.use_dsa:
            att = att + self.bias(log_p, mask)
        if not self.cfg.use_material_mixing:
            eye = torch.eye(S, dtype=torch.bool, device=x.device).view(1, 1, S, S)
            att = att.masked_fill(~eye, float('-inf'))
        keep = mask.view(B, 1, 1, S)
        att = att.masked_fill(~keep, float('-inf'))
        att = torch.nan_to_num(torch.softmax(att, dim=-1), nan=0.0)
        out = (att @ v).transpose(1, 2).reshape(B, S, d)
        return self.drop(self.proj(out))

class Block(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.n1 = nn.LayerNorm(d)
        self.att = MaterialSelfAttention(cfg)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))

    def forward(self, x, log_p, mask):
        x = x + self.att(self.n1(x), log_p, mask)
        x = x + self.ff(self.n2(x))
        return x

class BORAFormulaADR(nn.Module):

    def __init__(self, d_material: int, d_endpoint: int, n_mech: int, cfg: Config, n_endpoint: int=43, n_material: int=121):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        residual_dim = 2 * n_material if cfg.residual_on_composition else d
        self.adr_residual_w = nn.Parameter(torch.zeros(n_endpoint, residual_dim))
        self.adr_residual_b = nn.Parameter(torch.zeros(n_endpoint))
        self.label_mix = nn.Parameter(torch.zeros(2))
        self.label_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(cfg.label_graph_layers)])
        self.label_gate = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))

        def _cols(keep):
            if not keep:
                return None
            return torch.as_tensor([i for lo, hi in keep for i in range(int(lo), int(hi))], dtype=torch.long)
        self.register_buffer('mat_cols', _cols(cfg.mat_keep), persistent=False)
        self.register_buffer('adr_cols', _cols(cfg.adr_keep), persistent=False)
        if self.mat_cols is not None:
            d_material = int(self.mat_cols.numel())
        if self.adr_cols is not None:
            d_endpoint = int(self.adr_cols.numel())
        if (self.mat_cols is not None or self.adr_cols is not None) and cfg.use_multiview:
            raise ValueError('input-slice ablation and multiview are mutually exclusive')
        self.mat_views = [(0, 1024), (1024, d_material), (0, d_material)]
        self.adr_views = [(0, 1024), (1024, 1280), (1280, d_endpoint), (0, d_endpoint)]
        if not cfg.use_multiview:
            self.mat_views = [(0, d_material)]
            self.adr_views = [(0, d_endpoint)]
        self.mat_in = nn.ModuleList([nn.Sequential(nn.LayerNorm(b - a), nn.Linear(b - a, d), nn.GELU(), nn.Dropout(cfg.dropout)) for a, b in self.mat_views])
        self.adr_in = nn.ModuleList([nn.Sequential(nn.LayerNorm(b - a), nn.Linear(b - a, d), nn.GELU(), nn.Dropout(cfg.dropout)) for a, b in self.adr_views])
        self.dose_gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.head = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        self.mech = nn.Sequential(nn.LayerNorm(n_mech), nn.Linear(n_mech, 16), nn.GELU(), nn.Linear(16, 1))
        self.adr_bias = nn.Linear(d, 1)
        if cfg.use_shared_bilinear:
            self.shared_formula = nn.Linear(d, d, bias=False)
            self.shared_endpoint = nn.Linear(d, d, bias=False)
            self.shared_bilinear_scale = nn.Parameter(torch.tensor(0.1))
        self.n_support = 3
        self.n_channel = 4
        if cfg.use_support_fusion:
            self.fuse = nn.Sequential(nn.Linear(d + self.n_support, 32), nn.GELU(), nn.Linear(32, self.n_channel))
        self.channel_scale = nn.Parameter(torch.ones(self.n_channel))
        if cfg.use_channel_balance:
            self.register_buffer('channel_balance', torch.ones(self.n_channel))
        else:
            self.channel_balance = None
        if cfg.use_multiview:
            self.view_mix = nn.Parameter(torch.zeros(cfg.view_topk))
        self.mem_f = nn.Linear(d, d)
        self.mem_a = nn.Linear(d, d)
        self.log_tau_f = nn.Parameter(torch.tensor(float(cfg.mem_tau_init)))
        self.log_tau_a = nn.Parameter(torch.tensor(float(cfg.mem_tau_init)))
        self.mem_scale = nn.Parameter(torch.tensor(1.0))
        self.mem_bias = nn.Parameter(torch.tensor(0.0))
        if cfg.use_composition:
            self.comp_u = nn.Linear(2 * n_material, cfg.comp_rank, bias=False)
            self.comp_g = nn.Linear(d, cfg.comp_rank)
            self.comp_sparse = nn.Parameter(torch.zeros(2 * n_material, n_endpoint))
            self.comp_bias = nn.Parameter(torch.zeros(n_endpoint))

    def aggregate_views(self, view_logits: torch.Tensor) -> torch.Tensor:
        v = view_logits.shape[0]
        if not self.cfg.use_multiview or v == 1:
            return view_logits[0]
        k = min(int(self.cfg.view_topk), v)
        top, _ = torch.topk(view_logits, k, dim=0)
        w = self.view_mix[:k].unsqueeze(-1)
        return top.mean(0) + (w * top).sum(0)

    def composition_logit(self, presence, dose, z):
        total = (dose * (presence > 0)).sum(dim=-1, keepdim=True).clamp_min(EPS)
        simplex = dose / total
        x = torch.cat([presence, simplex], dim=-1)
        low = self.comp_u(x) @ self.comp_g(z).t()
        sparse = x @ self.comp_sparse if self.cfg.use_composition_sparse else torch.zeros_like(low)
        return low + sparse + self.comp_bias.unsqueeze(0)

    @staticmethod
    def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
        return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(EPS)

    def label_graph(self, z, hierarchy, labels, observed):
        mix = torch.softmax(self.label_mix, dim=0)
        assoc = mix[0] * hierarchy
        if labels is not None:
            oy = observed * labels
            co = oy.t() @ oy
            n = oy.sum(dim=0)
            cond = co / n.unsqueeze(0).clamp_min(1.0)
            cond = cond - torch.diag_embed(torch.diagonal(cond))
            assoc = assoc + mix[1] * cond
        adj = self._row_normalize(assoc.clamp_min(0.0))
        if labels is not None:
            n_pos = (observed * labels).sum(dim=0)
            evidence = torch.stack([torch.log1p(n_pos), (n_pos > 0).float()], dim=-1)
        else:
            evidence = torch.zeros(z.shape[0], 2, device=z.device)
        alpha = torch.sigmoid(self.label_gate(evidence))
        out = z
        for layer in self.label_proj:
            out = out + alpha * layer(adj @ out)
        return out

    def memory(self, u_query, u_ref, z, labels, observed, same_cohort: bool):

        def sparse_softmax(logits: torch.Tensor, k: int) -> torch.Tensor:
            if k <= 0 or k >= logits.shape[-1]:
                return torch.softmax(logits, dim=-1)
            cut = logits.topk(k, dim=-1).values[..., -1:]
            return torch.softmax(logits.masked_fill(logits < cut, float('-inf')), dim=-1)
        uq = F.normalize(self.mem_f(u_query), dim=-1)
        ur = F.normalize(self.mem_f(u_ref), dim=-1)
        za = F.normalize(self.mem_a(z), dim=-1)
        Sf = sparse_softmax(uq @ ur.t() / self.log_tau_f.exp().clamp(0.02, 10.0), self.cfg.mem_topk_f)
        Sa = sparse_softmax(za @ za.t() / self.log_tau_a.exp().clamp(0.02, 10.0), self.cfg.mem_topk_a)
        num = Sf @ (observed * labels) @ Sa.t()
        den = Sf @ observed @ Sa.t()
        if same_cohort:
            self_f = torch.diagonal(Sf).unsqueeze(1)
            self_a = torch.diagonal(Sa).unsqueeze(0)
            joint = self_f * self_a * observed
            num = num - joint * labels
            den = den - joint
        return (num / den.clamp_min(EPS), den, Sf)

    def encode_formula(self, mat_feat, dose, mask, view: int=-1):
        view = view if view >= 0 else len(self.mat_views) - 1
        a, b = self.mat_views[view]
        if self.mat_cols is not None:
            mat_feat = mat_feat[..., self.mat_cols]
        x = self.mat_in[view](mat_feat[..., a:b])
        w = dose * mask
        total = w.sum(dim=-1, keepdim=True).clamp_min(EPS)
        p = (w / total).clamp_min(EPS)
        log_p = torch.log(p)
        if self.cfg.use_dose_gate:
            order = torch.argsort(torch.argsort(-w, dim=-1), dim=-1).float()
            rank = order / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            feats = torch.stack([p, torch.log1p(dose.clamp_min(0.0)), rank], dim=-1)
            gate = F.softplus(self.dose_gate(feats).squeeze(-1))
        else:
            gate = torch.ones_like(p)
        gate = gate * mask
        for block in self.blocks:
            x = block(x, log_p, mask)
        return (self.norm(x), gate)

    def pooled_formula(self, mat_feat, dose, mask, view: int=-1):
        tokens, gate = self.encode_formula(mat_feat, dose, mask, view)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (tokens * gate.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, mat_feat, dose, mask, adr_feat, pair_mech, labels=None, observed=None, ref=None, hierarchy=None, presence=None, comp_dose=None, views=(-1, -1), balance_pairs=None):
        vf, va = views
        tokens, gate = self.encode_formula(mat_feat, dose, mask, vf)
        va = va if va >= 0 else len(self.adr_views) - 1
        a0, a1 = self.adr_views[va]
        if self.adr_cols is not None:
            adr_feat = adr_feat[..., self.adr_cols]
        z = self.adr_in[va](adr_feat[..., a0:a1])
        if self.cfg.use_label_graph and hierarchy is not None:
            z = self.label_graph(z, hierarchy, labels, observed)
        v = self.v_proj(tokens) * gate.unsqueeze(-1)
        k = self.k_proj(tokens)
        Fm, S, d = tokens.shape
        A = z.shape[0]
        if self.cfg.use_readout:
            q = self.q_proj(z)
            att = torch.einsum('ad,fsd->afs', q, k) / d ** 0.5
            att = att.masked_fill(~mask.unsqueeze(0), float('-inf'))
            att = torch.nan_to_num(torch.softmax(att, dim=-1), nan=0.0)
            pooled = torch.einsum('afs,fsd->afd', att, v)
        else:
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean = (v * mask.unsqueeze(-1)).sum(dim=1) / denom
            pooled = mean.unsqueeze(0).expand(A, Fm, d)
            att = torch.zeros(A, Fm, S, device=tokens.device)
        zz = z.unsqueeze(1).expand(A, Fm, d)
        struct = self.head(torch.cat([pooled, zz, pooled * zz], dim=-1)).squeeze(-1)
        if self.cfg.use_shared_bilinear:
            shared_f = F.normalize(self.shared_formula(pooled), dim=-1)
            shared_a = F.normalize(self.shared_endpoint(zz), dim=-1)
            struct = struct + self.shared_bilinear_scale * (shared_f * shared_a).sum(dim=-1)
        coverage = (mask.sum(dim=-1) / mask.shape[-1]).unsqueeze(0).expand(A, Fm)
        if self.cfg.use_memory and labels is not None:
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            u = (tokens * gate.unsqueeze(-1)).sum(dim=1) / denom
            if ref is None:
                u_ref, same = (u, True)
            else:
                u_ref, same = (self.pooled_formula(*ref, view=vf), False)
            mem, avail, Sf = self.memory(u, u_ref, z, labels, observed, same)
            mem_logit = self.mem_scale * torch.logit(mem.clamp(0.0001, 1 - 0.0001)) + self.mem_bias
            has_support = (avail > EPS).float()
            mem_term = (mem_logit * has_support).transpose(0, 1)
            log_avail = torch.log1p(avail.clamp_min(0.0)).transpose(0, 1)
            if same:
                nearest = (Sf - torch.diag_embed(torch.diagonal(Sf))).max(dim=-1).values
            else:
                nearest = Sf.max(dim=-1).values
            nearest = nearest.unsqueeze(0).expand(A, Fm)
        else:
            mem_term = torch.zeros(A, Fm, device=tokens.device)
            log_avail = torch.zeros(A, Fm, device=tokens.device)
            nearest = torch.zeros(A, Fm, device=tokens.device)
        support_feat = torch.stack([log_avail, nearest, coverage], dim=-1)
        if self.cfg.use_support_fusion:
            gates = torch.sigmoid(self.fuse(torch.cat([zz, support_feat], dim=-1)))
        else:
            gates = torch.full((A, Fm, self.n_channel), 0.5, device=tokens.device)
        weights = gates * self.channel_scale
        aux = {'struct': struct}
        m = torch.zeros_like(struct)
        if self.cfg.use_mech:
            m = self.mech(pair_mech).squeeze(-1).transpose(0, 1)
            aux['mech'] = m
        if self.cfg.use_memory and labels is not None:
            aux['memory'] = mem_term
        comp = torch.zeros_like(struct)
        if self.cfg.use_composition and presence is not None:
            comp = self.composition_logit(presence, comp_dose, z).transpose(0, 1)
            aux['composition'] = comp
        if self.cfg.use_channel_balance:
            channels = torch.stack([
                struct if self.cfg.use_struct else torch.zeros_like(struct),
                m,
                mem_term if self.cfg.use_memory and labels is not None else torch.zeros_like(struct),
                comp,
            ], dim=-1)
            if self.training and balance_pairs is not None:
                adr_index, formula_index = balance_pairs
                spread = channels[adr_index, formula_index].detach().std(dim=0, unbiased=False).clamp_min(1e-3)
                active = torch.as_tensor([
                    self.cfg.use_struct,
                    self.cfg.use_mech,
                    self.cfg.use_memory and labels is not None,
                    self.cfg.use_composition and presence is not None,
                ], device=spread.device, dtype=torch.bool)
                spread = torch.where(active, spread, torch.ones_like(spread))
                momentum = float(self.cfg.channel_balance_momentum)
                self.channel_balance.mul_(momentum).add_((1.0 - momentum) * spread)
            logit = (weights * channels / self.channel_balance.clamp_min(1e-3)).sum(dim=-1)
        else:
            logit = weights[..., 0] * struct if self.cfg.use_struct else torch.zeros_like(struct)
            if self.cfg.use_mech:
                logit = logit + weights[..., 1] * m
            if self.cfg.use_memory and labels is not None:
                logit = logit + weights[..., 2] * mem_term
            if self.cfg.use_composition and presence is not None:
                logit = logit + weights[..., 3] * comp
        if self.cfg.use_adr_residual:
            if self.cfg.residual_on_composition and presence is not None:
                total = (comp_dose * (presence > 0)).sum(dim=-1, keepdim=True).clamp_min(EPS)
                x = torch.cat([presence, comp_dose / total], dim=-1)
                logit = logit + (x @ self.adr_residual_w.t()).t() + self.adr_residual_b.unsqueeze(1)
            else:
                residual = (pooled * self.adr_residual_w.unsqueeze(1)).sum(-1) + self.adr_residual_b.unsqueeze(1)
                if self.cfg.residual_support_gate and observed is not None:
                    residual = residual * (observed.sum(dim=0) > EPS).float().unsqueeze(1)
                logit = logit + residual
        if self.cfg.use_adr_bias:
            logit = logit + self.adr_bias(z)
        aux = {k: v.transpose(0, 1) for k, v in aux.items()}
        return (logit.transpose(0, 1), att.transpose(0, 1), gate, aux)

    def penalty(self, gate: torch.Tensor) -> torch.Tensor:
        value = self.cfg.gate_l1 * gate.abs().mean()
        if self.cfg.use_composition:
            n_a = self.comp_sparse.shape[1]
            value = value + self.cfg.comp_l1 * self.comp_sparse.abs().sum() / n_a
            value = value + self.cfg.comp_l2 * self.comp_sparse.pow(2).sum() / n_a
        if self.cfg.use_adr_residual and self.cfg.residual_on_composition:
            n_a = self.adr_residual_w.shape[0]
            value = value + self.cfg.comp_l1 * self.adr_residual_w.abs().sum() / n_a
            value = value + self.cfg.comp_l2 * self.adr_residual_w.pow(2).sum() / n_a
        return value

    def param_groups(self, lr: float, weight_decay: float):
        comp = {'comp_sparse', 'comp_bias', 'adr_residual_w', 'adr_residual_b'}
        comp = {n for n in comp if any((n == a for a, _ in self.named_parameters()))}
        head, branch = ([], [])
        for name, param in self.named_parameters():
            (branch if name in comp else head).append(param)
        return [{'params': head, 'lr': lr, 'weight_decay': weight_decay}, {'params': branch, 'lr': self.cfg.comp_lr, 'weight_decay': 0.0}]
