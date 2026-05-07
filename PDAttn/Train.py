"""Training and validation loops for PDAttn (single epoch + optional metric pass).

Handles ``torch_geometric.nn.DataParallel`` by forwarding ``batch.to_data_list()`` when
needed. Validation runs ``tools.inference.evaluate`` and ``tools.assessment.run_assessment``
with stderr-only metric printing (stdout suppressed) to keep logs readable.
"""
from __future__ import annotations

import contextlib
import io

import torch

from tools.assessment import _compute_O_pert
from tools.inference import evaluate
from tools.utils import ProgressBar
from tools import assessment as ASM


def _unpack_pred(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    C_ctrl,
    device,
    epoch,
    print_every=50,
):
    """One training epoch with ``SystemaLoss`` (MSE + CDF term)."""
    model.train()
    total = 0.0

    try:
        from torch_geometric.nn import DataParallel as GeoDP

        is_geodp = isinstance(model, GeoDP)
    except Exception:
        is_geodp = False

    pbar = ProgressBar(total=len(train_loader), desc=f"Train Ep{epoch}")
    for step, batch in enumerate(train_loader, 1):
        batch = batch.to(device)
        optimizer.zero_grad()
        y = batch.y
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.dim() == 1:
            B = int(getattr(batch, "num_graphs", 1))
            G = y.numel() // B
            y = y.view(B, G)
        else:
            B = y.shape[0]
        if is_geodp:
            data_list = batch.to_data_list()
            outputs = model(data_list)
        else:
            outputs = model(batch)
        pred = _unpack_pred(outputs)

        ret = loss_fn(
            pred=pred,
            y_abs=y,
            C_ctrl=C_ctrl,
        )
        loss = ret["total"]
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        if (step % print_every) == 0:
            print(
                f"[Train] epoch={epoch} step={step}/{len(train_loader)} "
                f"total={ret['total'].item():.6f} "
                f"| mse={ret['mse'].item():.6f} | cdf={ret['cdf'].item():.6f}"
            )
        pbar.update(1)
    pbar.close()
    avg_loss = total / max(1, len(train_loader))

    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, val_loader, device, epoch, C_ctrl, pert=None):
    """Evaluate on ``val_loader``; print overall and all-top-20 DE metrics (English labels)."""
    model.eval()
    pbar = ProgressBar(total=len(val_loader), desc=f"Valid Ep{epoch}")

    try:
        results = evaluate(val_loader, model, uncertainty=False, device=device)

        O_pert = None
        if pert is not None:
            try:
                train_perts = None
                if (
                    hasattr(pert, "set2conditions")
                    and pert.set2conditions
                    and "train" in pert.set2conditions
                ):
                    train_perts = pert.set2conditions["train"]
                O_pert = _compute_O_pert(pert.adata, train_perts=train_perts)
                O_pert = torch.from_numpy(O_pert).float().to(device)
            except Exception as e:
                print(f"Could not compute O_pert, using None: {e}")
                O_pert = None

        with contextlib.redirect_stdout(io.StringIO()):
            _res = ASM.run_assessment(
                model=model,
                loaders={"test_loader": val_loader},
                device=device,
                C=C_ctrl,
                O_pert=O_pert,
                subgroup=None,
                results=results,
            )

        ov = _res["overall"]
        t20 = _res["all_top20"]

        print(
            f"[Val overall] mse={ov['mse']:.6f}  pearson={ov['pearson']:.6f}  "
            f"systema_pearson={ov['systema_pearson']:.6f}"
        )
        print(
            f"[Val all top-20 DE] mse_de={t20['mse_de']:.6f}  pearson_de={t20['pearson_de']:.6f}  "
            f"systema_pearson_de={t20['systema_pearson_de']:.6f}"
        )
        metrics = {"overall": ov, "all_top20": t20}

    finally:
        pbar.update(len(val_loader))
        pbar.close()

    return metrics
