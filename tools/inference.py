"""Model inference and batched evaluation utilities."""

import numpy as np
import torch

try:
    from tools.utils import ProgressBar
except Exception:

    class ProgressBar:
        def __init__(self, total, desc=""):
            self.n, self.total, self.desc = 0, int(total), desc

        def update(self, k):
            self.n += int(k)

        def close(self):
            pass


def _to_np(x):
    if x is None:
        return None
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _extract_pert_labels(batch, B):
    """Resolve perturbation labels (strings) per sample for subgroup alignment."""
    import numpy as np

    labels = None

    # Prefer batch.pert: list of names, e.g. "RRAS2+ctrl" for single/double/combo perturbations.
    if hasattr(batch, "pert"):
        p = getattr(batch, "pert")
        if isinstance(p, (list, tuple)):
            labels = [str(x) for x in p]
        else:
            try:
                arr = np.asarray(p)
                if arr.ndim == 1:
                    labels = [str(x) for x in arr.tolist()]
            except Exception:
                labels = None

    # Else batch.pert_cat: strings, object array, or ID matrix for combo perturbations.
    if labels is None and hasattr(batch, "pert_cat"):
        pc = getattr(batch, "pert_cat")
        if isinstance(pc, (list, tuple)):
            labels = [str(x) for x in pc]
        else:
            try:
                arr = np.asarray(pc)
                if arr.ndim == 1:
                    labels = [str(x) for x in arr.tolist()]
                elif arr.ndim == 2:
                    labels = [
                        "+".join(
                            [
                                str(int(x)) if isinstance(x, (int, np.integer)) else str(x)
                                for x in row
                                if (str(x) != "" and str(x) != "-1")
                            ]
                        )
                        or "ctrl"
                        for row in arr
                    ]
            except Exception:
                labels = None

    # Fallback: condition, pert_name, pert_cat tensor, or pert_idx.
    if labels is None and hasattr(batch, "condition"):
        try:
            arr = np.asarray(getattr(batch, "condition"))
            labels = [str(x) for x in arr.reshape(-1).tolist()]
        except Exception:
            pass
    if labels is None and hasattr(batch, "pert_name"):
        pn = batch.pert_name
        labels = list(pn) if isinstance(pn, (list, tuple)) else [str(pn)] * B
    if labels is None and hasattr(batch, "pert_cat") and isinstance(batch.pert_cat, torch.Tensor):
        arr = batch.pert_cat.detach().cpu().numpy()
        if arr.ndim == 1:
            labels = [f"id{int(x)}" for x in arr.tolist()]
        else:
            labels = []
            for row in arr:
                ids = [f"id{int(x)}" for x in row.tolist() if int(x) >= 0]
                labels.append("+".join(ids) if ids else "ctrl")
    if labels is None and hasattr(batch, "pert_idx"):
        pi = batch.pert_idx
        if isinstance(pi, torch.Tensor):
            if pi.dim() == 1:
                labels = [f"id{int(x)}" for x in pi.detach().cpu().tolist()]
            else:
                labels = []
                for row in pi.detach().cpu().tolist():
                    ids = [f"id{int(x)}" for x in row if int(x) >= 0]
                    labels.append("+".join(ids) if ids else "ctrl")
        elif isinstance(pi, (list, tuple)):
            if len(pi) and isinstance(pi[0], (list, tuple)):
                labels = []
                for row in pi:
                    ids = [f"id{int(x)}" for x in row if int(x) >= 0]
                    labels.append("+".join(ids) if ids else "ctrl")
            else:
                labels = [f"id{int(x)}" for x in pi]

    if labels is None or len(labels) != B:
        labels = ["unknown"] * B
    return labels


def evaluate(loader, model, uncertainty: bool, device: torch.device, desc: str = "[eval]"):
    """Run the model on a loader; collect predictions and targets as [B, G]; keep per-sample labels.

    Supports torch_geometric.nn.DataParallel (GeoDP), which expects a list[Data] per step.
    """
    model.eval()
    pred, truth, pert_cat = [], [], []
    pred_de, truth_de, de_idx_list = [], [], []
    logvar = []

    try:
        from torch_geometric.nn import DataParallel as GeoDP

        is_geodp = isinstance(model, GeoDP)
    except Exception:
        is_geodp = False

    pbar = ProgressBar(total=len(loader), desc=desc)

    for step, batch in enumerate(loader, start=1):
        # Match training: move batch to device, then optionally split for GeoDP.
        batch = batch.to(device)
        B = getattr(batch, "num_graphs", 1)

        with torch.no_grad():
            inputs = batch.to_data_list() if is_geodp else batch
            out = model(inputs)
            unc = None
            if isinstance(out, (tuple, list)):
                if len(out) == 0:
                    raise RuntimeError("Model forward returned an empty tuple.")
                yhat = out[0]
                if uncertainty and len(out) >= 3 and isinstance(out[2], dict):
                    unc = out[2].get("logvar", None)
                if unc is not None:
                    lv = _to_np(unc)
                    if lv is not None:
                        logvar.append(lv)
            else:
                yhat = out
            y = batch.y

        if isinstance(y, (tuple, list)):
            y = y[0]
        if isinstance(yhat, (tuple, list)):
            yhat = yhat[0]
        if y.dim() == 1:
            G = y.numel() // B
            y = y.view(B, G)
        if yhat.dim() == 1:
            G = yhat.numel() // B
            yhat = yhat.view(B, G)

        pred.append(_to_np(yhat))
        truth.append(_to_np(y))

        labels = _extract_pert_labels(batch, B)
        pert_cat.extend(labels)

        pred_de.append([])
        truth_de.append([])
        de_idx_list.append([])

        pbar.update(1)
    pbar.close()

    results = {
        "pred": np.concatenate(pred, axis=0) if len(pred) else np.zeros((0, 0)),
        "truth": np.concatenate(truth, axis=0) if len(truth) else np.zeros((0, 0)),
        "pert_cat": np.asarray(pert_cat),
        "pred_de": pred_de,
        "truth_de": truth_de,
        "de_idx": de_idx_list,
        "logvar": (np.concatenate(logvar, axis=0) if len(logvar) else None),
    }
    return results
