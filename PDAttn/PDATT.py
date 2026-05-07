"""High-level PDAttn factory and training-set statistics (control / perturbation centroids).

``build_pdattn_model`` wires ``GenePertGraphBuilder.make_model_args`` into ``PDAttnModel``.
``estimate_ctrl_and_pert_centroids`` scans the train loader to obtain a global control
centroid ``C_ctrl`` and a mean of per-perturbation centroids ``mu_ref`` (useful as an
auxiliary reference; Systema-style ``O_pert`` for metrics is computed separately in
``tools.assessment``).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from PDAttn.M_model import PDAttnModel


def build_pdattn_model(
    builder,
    decoder_hidden_size: int = 16,
    hidden_dim: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    dropout: Optional[float] = None,
) -> nn.Module:
    """Instantiate ``PDAttnModel`` from a ``GenePertGraphBuilder`` and optional overrides.

    Defaults: hidden 64, 4 heads, dropout 0.1.
    """
    hidden_size = int(hidden_dim) if hidden_dim is not None else 64
    num_heads = int(num_attention_heads) if num_attention_heads is not None else 4
    drop_rate = float(dropout) if dropout is not None else 0.1
    args: Dict = builder.make_model_args(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        decoder_hidden_size=int(decoder_hidden_size),
    )
    args["dropout"] = drop_rate
    args["hidden_size"] = hidden_size
    args["num_attention_heads"] = num_heads
    args["decoder_hidden_size"] = int(decoder_hidden_size)
    model = PDAttnModel(args)

    return model


@torch.no_grad()
def estimate_ctrl_and_pert_centroids(
    train_loader,
    device: torch.device,
    ctrl_name: str = "ctrl",
    max_batches: Optional[int] = None,
    verbose: bool = True,
):
    """Compute batch statistics from **training** cells only.

    Returns:
        C_ctrl: mean expression over all control-labeled rows, shape ``[G]``.
        mu_ref: mean of per-perturbation empirical means (one centroid per non-control
            condition), shape ``[G]``. If there are no non-control rows, ``mu_ref`` equals
            ``C_ctrl``.

    Batch fields: uses ``batch.y`` as expression and ``batch.pert`` or ``batch.condition``
    as string labels.
    """
    sums_ctrl, n_ctrl = None, 0
    per_pert_sum, per_pert_cnt = {}, {}

    total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
    if max_batches is not None and total_batches is not None:
        total_batches = min(max_batches, total_batches)

    if verbose:
        print(f"Scanning train loader (up to {total_batches or 'all'} batches)...")

    for idx, batch in enumerate(train_loader):
        if max_batches is not None and idx >= max_batches:
            break
        batch = batch.to(device)
        y = batch.y
        if isinstance(y, (list, tuple)):
            y = y[0]
        if y.dim() == 1:
            B = getattr(batch, "num_graphs", 1)
            G = y.numel() // B
            y = y.view(B, G)
        B, G = y.shape

        if hasattr(batch, "pert"):
            cond = batch.pert
        elif hasattr(batch, "condition"):
            cond = batch.condition
        else:
            cond = ["unknown"] * B
        cond = [
            str(c) if not isinstance(c, (bytes, bytearray)) else c.decode("utf-8")
            for c in (cond if isinstance(cond, (list, tuple)) else list(cond))
        ]

        if sums_ctrl is None:
            sums_ctrl = torch.zeros(G, device=device)

        for ci, yi in zip(cond, y):
            if ci == ctrl_name:
                sums_ctrl += yi
                n_ctrl += 1
            else:
                if ci not in per_pert_sum:
                    per_pert_sum[ci] = torch.zeros(G, device=device)
                    per_pert_cnt[ci] = 0
                per_pert_sum[ci] += yi
                per_pert_cnt[ci] += 1

    if n_ctrl == 0:
        raise RuntimeError(
            f"No control-labeled samples (ctrl_name='{ctrl_name}') in train_loader; cannot estimate C_ctrl."
        )
    C_ctrl = (sums_ctrl / float(n_ctrl)).detach()

    if len(per_pert_sum) == 0:
        mu_ref = C_ctrl.clone()
    else:
        centroids = [(per_pert_sum[p] / float(per_pert_cnt[p])) for p in per_pert_sum.keys()]
        mu_ref = torch.stack(centroids, dim=0).mean(dim=0)

    if verbose:
        print(
            f"Centroids done: C_ctrl {tuple(C_ctrl.shape)}, mu_ref {tuple(mu_ref.shape)}, "
            f"n_ctrl={n_ctrl}, n_perts={len(per_pert_sum)}"
        )

    return C_ctrl.detach(), mu_ref.detach()
