"""PDAttn: perturbation-aware gene expression modeling with dual encoders.

Architecture
-------------------------
1. **Gene branch** — Each gene has a learned embedding; observed control expression is
   linearly projected and combined with that prior. A *linear* (kernelized) multi-head
   self-attention over the gene dimension mixes information without an explicit gene–gene
   graph (feature map :math:`\\phi(x)=\\mathrm{ELU}(x)+1` for efficient low-rank attention).

2. **Perturbation branch** — Perturbation embeddings are refined with **SGConv** layers on
   the GO-derived perturbation graph (edges and weights supplied by ``GenePertGraphBuilder``).

3. **Conditioning** — **FiLM** (scale ``gamma`` and shift ``beta`` from fused perturbation
   features) modulates per-gene hidden states. A second linear attention module treats a
   compressed cell vector fused with perturbation summary as a length-1 key/value context
   (perturbation–cell cross-attention over genes).

4. **Readout** — A shared per-gene linear map produces a residual **delta** ``de`` added
   to input expression to predict post-perturbation profiles ``pred = x + de``. For
   multi-perturbation (combo) cells, an additional per-gene residual head is gated on when
   more than one perturbation index is active.

Optional flags on ``forward`` expose intermediate tensors for analysis or visualization
(see method docstring).

Inputs are **PyG** ``Data`` / batched graphs: expression in ``x`` (or ``expr`` / ``X``),
perturbation identity via ``pert_mask`` or ``pert_idx``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SGConv


def _relu(x):
    return torch.relu(x)


class LinearAttention(nn.Module):
    """Efficient attention via nonnegative feature map φ(x) = ELU(x) + 1 (linear-time in sequence length).

    Implements the scaling pattern ``(φ(Q) @ (φ(K)^T V)) / (φ(Q) @ sum_k φ(K_k))`` so complexity is
    O(N) in the number of positions for fixed head dimension. This is not full softmax attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias_lambda: float = 1.0,
        bias_rmin: float = 0.1,
        learnable_distance: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        self.learnable_distance = bool(learnable_distance)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.bias_lambda = float(bias_lambda)
        self.bias_rmin = float(bias_rmin)

    def _feature_map(self, x: Tensor) -> Tensor:
        """Nonnegative map for kernelized attention normalization."""
        return F.elu(x) + 1.0

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply linear attention.

        Shapes:
            query: ``[B, N_q, d]`` or ``[N_q, d]``
            key, value: ``[B, N_k, d]`` or ``[N_k, d]``
        Returns:
            Same rank as input query (batch dims preserved).
        """
        squeeze_out = False
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_out = True

        B, N_q, _ = query.shape
        _, N_k, _ = key.shape

        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)

        Q_m = self._feature_map(Q)
        K_m = self._feature_map(K)

        KV = torch.einsum("bhkd,bhke->bhde", K_m, V)
        attn_out = torch.einsum("bhqd,bhde->bhqe", Q_m, KV)

        K_sum = K_m.sum(dim=2)
        normalizer = torch.einsum("bhqd,bhd->bhq", Q_m, K_sum).unsqueeze(-1).clamp_(min=1e-8)
        attn_out = attn_out / normalizer

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        output = self.out_proj(attn_out)
        output = self.dropout(output)

        if squeeze_out:
            output = output.squeeze(0)

        return output


class MLP(nn.Module):
    """Fully connected stack with optional batch normalization (training-safe for batch size 1)."""

    def __init__(self, sizes: List[int], use_bn: bool = True, last_act: str = "linear"):
        super().__init__()
        self.use_bn = bool(use_bn)
        self._last_act = last_act
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                if self.use_bn:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        for m in self.net:
            if isinstance(m, nn.BatchNorm1d):
                if self.training and x.dim() == 2 and x.size(0) == 1:
                    x = F.batch_norm(
                        x,
                        m.running_mean,
                        m.running_var,
                        m.weight,
                        m.bias,
                        training=False,
                        momentum=m.momentum,
                        eps=m.eps,
                    )
                else:
                    x = m(x)
            else:
                x = m(x)
        if self._last_act == "relu":
            return F.relu(x, inplace=False)
        return x


class PDAttnModel(nn.Module):
    """Predict perturbed expression as residual ``pred = x + de`` given control-like input ``x``."""

    def __init__(self, args: Dict):
        super().__init__()
        self.args = args
        self.G = int(args["num_genes"])
        self.P = int(args["num_perts"])
        self.d = int(args.get("hidden_size", 64))
        self.n_go_layers = int(args.get("num_go_gnn_layers", 1))
        self.num_heads = int(args.get("num_attention_heads", 4))
        self.dec_hidden = int(args.get("decoder_hidden_size", 16))
        self.no_perturb = bool(args.get("no_perturb", False))
        self.dropout = float(args.get("dropout", 0.1))

        self.register_buffer("G_go", args["G_go"].long().contiguous())
        self.register_buffer("G_go_weight", args["G_go_weight"].float().contiguous())

        p2g = args.get("pert_to_gene", torch.zeros(self.P, self.G))
        if p2g.dim() == 1:
            mat = torch.zeros(self.P, self.G, dtype=torch.float32)
            idx = p2g.long()
            ok = (idx >= 0) & (idx < self.G)
            if ok.any():
                mat[torch.arange(self.P)[ok], idx[ok]] = 1.0
            p2g = mat
        self.register_buffer("pert_to_gene", p2g.float().contiguous(), persistent=False)

        self.gene_emb = nn.Embedding(self.G, self.d)
        self.expr_proj = nn.Linear(1, self.d)

        learnable_distance = bool(args.get("learnable_distance", True))
        bias_lambda = float(args.get("bias_lambda", 1.0))
        bias_rmin = float(args.get("bias_rmin", 0.1))

        self.attn_base = LinearAttention(
            self.d,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias_lambda=bias_lambda,
            bias_rmin=bias_rmin,
            learnable_distance=learnable_distance,
        )

        self.cell_compressor = MLP(
            sizes=[self.G, self.d * 2, self.d],
            use_bn=True,
            last_act="relu",
        )

        self.pert_cell_fuse = nn.Sequential(
            nn.Linear(2 * self.d, self.d),
            nn.LayerNorm(self.d),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.cross_attn_pert_cell = LinearAttention(
            self.d,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias_lambda=bias_lambda,
            bias_rmin=bias_rmin,
            learnable_distance=learnable_distance,
        )

        self.film_gamma = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.LayerNorm(self.d),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, self.d),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.LayerNorm(self.d),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, self.d),
        )

        self.pert_emb = nn.Embedding(self.P, self.d)
        self.go_layers_pert = nn.ModuleList(
            [SGConv(self.d, self.d, K=1) for _ in range(self.n_go_layers)]
        )
        self.pert_fuse = MLP([self.d, self.d, self.d], use_bn=False, last_act="relu")
        nn.init.xavier_normal_(self.gene_emb.weight)
        nn.init.xavier_normal_(self.pert_emb.weight)

        self.w = nn.Parameter(torch.empty(1, self.G, self.d))
        self.b = nn.Parameter(torch.zeros(1, self.G))
        nn.init.xavier_normal_(self.w)
        nn.init.xavier_normal_(self.b)
        self.dec_act = nn.ReLU()

        self.combo_de_head = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.LayerNorm(self.d),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, 1),
        )
        self.combo_de_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @staticmethod
    def _parse_x(data) -> Tensor:
        x = None
        for k in ("x", "expr", "X"):
            if hasattr(data, k):
                x = getattr(data, k).float()
                break
        if x is None:
            raise AttributeError("Batch must provide expression as x, expr, or X.")
        B = getattr(data, "num_graphs", None)
        if B is None and hasattr(data, "y") and torch.is_tensor(data.y):
            y = data.y
            if y.dim() == 2:
                B = y.size(0)
        if B is None:
            B = 1
        if x.dim() == 1:
            x = x.view(B, -1)
        elif x.dim() == 2 and x.size(0) != B and x.numel() % B == 0:
            x = x.view(B, -1)
        return x

    @staticmethod
    def _parse_pert_mask(data, P: int, B: int, dev) -> torch.Tensor:
        """Build a binary ``[B, P]`` mask of active perturbation indices per cell.

        Prefer ``data.pert_mask`` when present; otherwise scatter ones from ``data.pert_idx``
        (tensor, numpy, or nested lists; supports one or multiple indices per sample).
        """
        if hasattr(data, "pert_mask"):
            m = getattr(data, "pert_mask")
            if not torch.is_tensor(m):
                m = torch.as_tensor(m)
            m = m.to(dev).float()
            if m.dim() != 2 or m.size(0) != B or m.size(1) != P:
                if m.numel() == B * P:
                    m = m.view(B, P)
                else:
                    raise RuntimeError(
                        f"Invalid pert_mask shape {tuple(m.shape)}; expected [{B}, {P}]."
                    )
            return m

        if hasattr(data, "pert_idx"):
            raw = getattr(data, "pert_idx")
            ids: Optional[Tensor] = None

            if torch.is_tensor(raw):
                ids = raw
            else:
                try:
                    ids = torch.as_tensor(raw)
                except Exception:
                    ids = None

            if ids is not None:
                ids = ids.to(dev)
                if ids.ndim == 0:
                    ids = ids.view(1, 1).expand(B, 1)
                elif ids.ndim == 1:
                    if ids.numel() != B:
                        if ids.numel() % B == 0:
                            ids = ids.view(B, -1)
                        else:
                            raise RuntimeError(
                                f"pert_idx length {ids.numel()} incompatible with batch B={B}."
                            )
                    else:
                        ids = ids.view(B, 1)
                elif ids.ndim >= 2:
                    ids = ids.view(B, -1)
                ids = ids.long()
            else:

                def _is_seq(x) -> bool:
                    try:
                        len(x)
                        return not isinstance(x, (str, bytes))
                    except TypeError:
                        return False

                if not raw:
                    return torch.zeros((B, P), dtype=torch.float32, device=dev)

                if not any(_is_seq(e) for e in raw):
                    ids = torch.tensor(list(raw), device=dev, dtype=torch.long).view(B, 1)
                else:
                    L = max((len(e) if _is_seq(e) else 1) for e in raw)
                    ids = torch.full((B, L), -1, dtype=torch.long, device=dev)
                    for b, arr in enumerate(raw):
                        if _is_seq(arr):
                            for j, pid in enumerate(arr[:L]):
                                try:
                                    ids[b, j] = int(pid)
                                except Exception:
                                    pass
                        else:
                            try:
                                ids[b, 0] = int(arr)
                            except Exception:
                                pass

            m = torch.zeros((B, P), dtype=torch.float32, device=dev)
            if ids is not None and ids.numel() > 0:
                for b in range(ids.size(0)):
                    for j in range(ids.size(1)):
                        pid = int(ids[b, j].item())
                        if 0 <= pid < P:
                            m[b, pid] = 1.0
            return m

        return torch.zeros((B, P), dtype=torch.float32, device=dev)

    def forward(
        self,
        data,
        return_cross_attn_repr: bool = False,
        return_readout_decomposition: bool = False,
        return_umap_aux: bool = False,
    ) -> Union[
        Tensor,
        tuple,
    ]:
        """Predict post-perturbation expression ``[B, G]`` as ``x + de``.

        Args:
            data: PyG ``Data`` / ``Batch`` with expression and perturbation fields.
            return_cross_attn_repr: If True, also return ``(z_pre, z_post)`` pooled gene
                states before/after cross-attention, each ``[B, d]``.
            return_readout_decomposition: If True, return
                ``(pred, h_attn_base, h_gene, cross_delta)`` with gene-level tensors
                ``[B, G, d]`` (FiLM/cross-attn decomposition; scalar readout uses ``w``, ``b``;
                excludes combo-gated residual).
            return_umap_aux: If True, return
                ``(pred, x, cell_feature, ctx, h_gene, h_ctx)`` for visualization.

        Returns:
            Default: ``pred`` only, shape ``[B, G]``.
        """
        if self.no_perturb:
            x = self._parse_x(data)
            out = torch.zeros_like(x)
            B = x.size(0)
            if return_readout_decomposition:
                z = torch.zeros(B, self.G, self.d, device=x.device, dtype=x.dtype)
                return out, z, z, z
            if return_umap_aux:
                zd = torch.zeros(B, self.d, device=x.device, dtype=x.dtype)
                Gb = int(x.size(1)) if x.dim() == 2 else self.G
                zg = torch.zeros(B, Gb, self.d, device=x.device, dtype=x.dtype)
                return out, x, zd, zd, zg, zg
            if return_cross_attn_repr:
                z1 = torch.zeros(B, self.d, device=x.device, dtype=x.dtype)
                return out, z1, z1
            return out

        dev = self.gene_emb.weight.device
        x = self._parse_x(data).to(dev)
        B, G = x.shape
        if G != self.G:
            if x.numel() % self.G == 0:
                B = x.numel() // self.G
                x = x.view(B, self.G)
            else:
                raise AssertionError(f"Gene dimension {G} does not match model G={self.G}.")

        ids_g = torch.arange(self.G, device=dev, dtype=torch.long)
        h_base = self.gene_emb(ids_g)
        e = self.expr_proj(x.unsqueeze(-1))
        e = F.layer_norm(e, (self.d,))
        alpha_q, alpha_k, beta_v = 0.5, 0.5, 0.5
        H0 = h_base.unsqueeze(0).expand(B, -1, -1)
        Q0 = H0 + alpha_q * e
        K0 = H0 + alpha_k * e
        V0 = H0 + beta_v * e
        h_attn_base = self.attn_base(Q0, K0, V0)
        h_gene = h_attn_base

        ids_p = torch.arange(self.P, device=dev, dtype=torch.long)
        z_pert = self.pert_emb(ids_p)
        for i, layer in enumerate(self.go_layers_pert):
            z_pert = layer(z_pert, self.G_go, self.G_go_weight)
            if i < len(self.go_layers_pert) - 1:
                z_pert = torch.relu(z_pert)
        z_pert = z_pert.unsqueeze(0).expand(B, -1, -1)

        pert_mask = self._parse_pert_mask(data, self.P, B, dev)
        den = pert_mask.sum(dim=1, keepdim=True).clamp(min=1).to(z_pert.dtype)
        z_fused = (z_pert * pert_mask.unsqueeze(-1)).sum(dim=1) / den
        z_fused_expanded = z_fused.unsqueeze(1).expand(-1, self.G, -1)

        cell_feature = self.cell_compressor(x)
        ctx = self.pert_cell_fuse(torch.cat([z_fused, cell_feature], dim=-1))
        ctx_kv = ctx.unsqueeze(1)

        gamma = self.film_gamma(z_fused_expanded)
        beta = self.film_beta(z_fused_expanded)
        h_gene = gamma * h_gene + beta

        cross_delta = self.cross_attn_pert_cell(h_gene, ctx_kv, ctx_kv)
        h_ctx = h_gene + cross_delta

        z_pre = h_gene.mean(dim=1)
        z_post = h_ctx.mean(dim=1)

        de = (h_ctx * self.w).sum(dim=-1) + self.b.view(1, self.G)

        is_combo = (pert_mask.sum(dim=1) > 1.0).to(dtype=torch.float32)
        g_combo = is_combo.view(B, 1)
        delta_de = torch.tanh(self.combo_de_head(h_ctx).squeeze(-1)) * self.combo_de_scale
        de = de + g_combo * delta_de

        pred = x + de

        if return_readout_decomposition:
            return pred, h_attn_base, h_gene, cross_delta
        if return_umap_aux:
            return pred, x, cell_feature, ctx, h_gene, h_ctx
        if return_cross_attn_repr:
            return pred, z_pre, z_post
        return pred
