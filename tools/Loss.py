from __future__ import annotations

import torch
import torch.nn as nn


class SystemaLoss(nn.Module):
    """Composite loss: MSE between predictions and targets plus a CDF-matching term."""

    def __init__(
        self,
        enable_mse: bool = True,
        enable_cdf: bool = True,
        cdf_lambda: float = 0.2,
        cdf_p: int = 2,
        zero_eps: float = 1e-8,
    ):
        super().__init__()
        self.enable_mse = bool(enable_mse)
        self.enable_cdf = bool(enable_cdf)
        self.cdf_lambda = float(cdf_lambda)
        self.cdf_p = int(cdf_p)
        self.zero_eps = float(zero_eps)

        self.register_buffer("mu_ref", None, persistent=False)
        self.register_buffer("delta_sys_mean", None, persistent=False)

    @torch.no_grad()
    def set_systema_refs(self, mu_ref, delta_sys_mean, s_vec=None):
        self.mu_ref = mu_ref.detach() if mu_ref is not None else None
        self.delta_sys_mean = (
            delta_sys_mean.detach() if delta_sys_mean is not None else None
        )

    def forward(
        self,
        pred: torch.Tensor,
        y_abs: torch.Tensor,
        C_ctrl: torch.Tensor,
    ):
        C_ctrl = C_ctrl.float().view(1, -1)

        pred_abs = pred
        delta_true = y_abs - C_ctrl
        pred_delta = pred_abs - C_ctrl

        mse = (
            torch.mean((pred_abs - y_abs) ** 2)
            if self.enable_mse
            else torch.zeros((), device=pred_abs.device, dtype=pred_abs.dtype)
        )

        if self.enable_cdf:
            a_sorted, _ = torch.sort(pred_delta, dim=-1)
            b_sorted, _ = torch.sort(delta_true, dim=-1)
            diff = (a_sorted - b_sorted).abs()
            if self.cdf_p != 1:
                diff = diff.pow(float(self.cdf_p))
            cdf_val = diff.mean(dim=-1).mean()
            cdf_w = self.cdf_lambda * cdf_val
        else:
            cdf_w = torch.zeros((), device=pred_abs.device, dtype=pred_abs.dtype)

        total = mse + cdf_w

        return {
            "total": total,
            "mse": mse,
            "cdf": cdf_w,
        }
