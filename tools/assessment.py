"""Evaluation metrics for perturbation response prediction (PDAttn).

This module aggregates predictions from :func:`tools.inference.evaluate` and reports:

**GEARS-style metrics** (perturbation-wise centroid deltas w.r.t. the control centroid,
subgroup splits, top-DE-gene subsets, normalized MSE vs. a no-change baseline). The
definitions follow the evaluation conventions used in GEARS and related single-cell
perturbation benchmarks:

  https://github.com/snap-stanford/GEARS

**Systema-style metrics** (Pearson on deltas w.r.t. the perturbed-cell reference
``O_pert``, i.e. the mean of per-perturbation centroids over training non-control
conditions), as described in the Systema framework:

  https://github.com/mlbio-epfl/systema

``systema_pearson`` and ``systema_pearson_de`` use this reference. Other Pearson and
MSE terms are deltas w.r.t. the control centroid (GEARS-style).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from tools.inference import evaluate


def fmt6(x):
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _to_np(x):
    """Convert torch tensors, lists, or None to numpy arrays where applicable."""
    if x is None:
        return None
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _mse(a, b):
    a = _to_np(a).reshape(-1)
    b = _to_np(b).reshape(-1)
    if a.size == 0 or b.size == 0:
        return np.nan
    d = a - b
    return float(np.mean(d * d))


def _pearson(a, b):
    a = _to_np(a).reshape(-1)
    b = _to_np(b).reshape(-1)
    if a.size == 0 or b.size == 0:
        return np.nan
    # Constant vectors yield nan from pearsonr; leave as nan for downstream averaging.
    if np.allclose(a, a.mean()) or np.allclose(b, b.mean()):
        return np.nan
    v = pearsonr(a, b)[0]
    return float(v) if np.isfinite(v) else np.nan


def _group_mean_by_pert(pred, truth, pert_cat):
    """Average predictions and truths to per-perturbation centroids (single/d combo via string labels)."""
    pred = _to_np(pred)
    truth = _to_np(truth)
    pert_cat = _to_np(pert_cat)
    mu_p, mu_t = {}, {}
    perts = np.unique(pert_cat)
    for p in perts:
        idx = np.where(pert_cat == p)[0]
        if idx.size == 0:
            continue
        mu_p[p] = pred[idx].mean(axis=0)
        mu_t[p] = truth[idx].mean(axis=0)
    return mu_p, mu_t


def _avg_over_perts(mu_p, mu_t, ref_vec):
    """Per perturbation: MSE and Pearson between (mu - ref) for pred and truth; then mean over perts."""
    ref = _to_np(ref_vec).reshape(-1)
    pears, mses = [], []
    for p in mu_p.keys():
        dp = _to_np(mu_p[p]) - ref
        dt = _to_np(mu_t[p]) - ref
        pears.append(_pearson(dp, dt))
        mses.append(_mse(dp, dt))
    pears = [v for v in pears if np.isfinite(v)]
    mses = [v for v in mses if np.isfinite(v)]
    return (
        float(np.mean(mses)) if mses else np.nan,
        float(np.mean(pears)) if pears else np.nan,
    )


def _systema_pearson(pred, truth, pert_cat, O_pert):
    """Systema-Pearson: mean over p of corr(mu_pred[p] - O_pert, mu_truth[p] - O_pert).

    See Systema (mlbio-epfl/systema) for the perturbed-centroid reference formulation.
    """
    if O_pert is None:
        return np.nan
    O_pert = _to_np(O_pert).reshape(-1)
    mu_p, mu_t = _group_mean_by_pert(pred, truth, pert_cat)
    if len(mu_p) == 0:
        return np.nan
    vals = []
    for p in mu_p.keys():
        v = _pearson(_to_np(mu_p[p]) - O_pert, _to_np(mu_t[p]) - O_pert)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if len(vals) else np.nan


def _overall_metrics(results, C, O_pert):
    """Overall metrics on test perturbation centroids.

    - Pearson / MSE: deltas vs. control centroid C (GEARS-style).
    - systema_pearson: deltas vs. O_pert (Systema-style reference).
    - mse_norm: MSE_delta divided by baseline MSE when prediction is always C (zero delta).
    """
    pred = results["pred"]
    truth = results["truth"]
    perts = results["pert_cat"]

    mu_p, mu_t = _group_mean_by_pert(pred, truth, perts)

    mse_delta, pear_delta = _avg_over_perts(mu_p, mu_t, C)
    sys_pear = _systema_pearson(pred, truth, perts, O_pert)

    if len(mu_t) == 0:
        mse_baseline = np.nan
    else:
        mu_p_baseline = {p: _to_np(C) for p in mu_t.keys()}
        mse_baseline, _ = _avg_over_perts(mu_p_baseline, mu_t, C)

    if np.isfinite(mse_delta) and np.isfinite(mse_baseline) and mse_baseline > 0:
        mse_norm = float(mse_delta / mse_baseline)
    else:
        mse_norm = np.nan

    return {
        "mse": mse_delta,
        "mse_norm": mse_norm,
        "pearson": pear_delta,
        "systema_pearson": sys_pear,
    }


def _subset_mask(pert_cat, subgroup_list):
    """Boolean mask of samples whose perturbation label lies in subgroup_list."""
    pert_cat = _to_np(pert_cat)

    if subgroup_list is None:
        return np.ones_like(pert_cat, dtype=bool)

    try:
        if isinstance(subgroup_list, str):
            s = {subgroup_list}
        else:
            s = set(list(subgroup_list))
    except Exception:
        return np.zeros_like(pert_cat, dtype=bool)

    if len(s) == 0:
        return np.zeros_like(pert_cat, dtype=bool)

    return np.array([p in s for p in pert_cat], dtype=bool)


def _subset_metrics(results, C, O_pert, subgroup_list):
    mask = _subset_mask(results["pert_cat"], subgroup_list)
    if mask is None or not mask.any():
        return {
            "mse": np.nan,
            "mse_norm": np.nan,
            "pearson": np.nan,
            "systema_pearson": np.nan,
        }
    sub_results = {
        "pred": _to_np(results["pred"])[mask],
        "truth": _to_np(results["truth"])[mask],
        "pert_cat": _to_np(results["pert_cat"])[mask],
        "pred_de": [],
        "truth_de": [],
        "de_idx": [],
        "logvar": None,
    }
    return _overall_metrics(sub_results, C, O_pert)


def _subset_top20(results, C, O_pert, subgroup_list, adata=None):
    """Top-20 DE gene metrics (GEARS-style DE lists in ``adata.uns``).

    Steps:
      1. Aggregate to per-perturbation centroids.
      2. Resolve DE genes from ``adata.uns`` (same priority as GEARS):
         ``top_non_dropout_de_20``, then ``top_non_zero_de_20``, then
         ``rank_genes_groups_cov_all`` (first 20).
      3. On those indices, MSE and Pearson on deltas vs. C; Systema-Pearson uses O_pert.
      4. Average over perturbations; report normalized MSE vs. zero-delta baseline.
    """
    mask = _subset_mask(results["pert_cat"], subgroup_list)
    if mask is None or not mask.any():
        return {
            "mse_de": np.nan,
            "mse_de_norm": np.nan,
            "pearson_de": np.nan,
            "systema_pearson_de": np.nan,
        }

    PRED = _to_np(results["pred"])[mask]
    TRUTH = _to_np(results["truth"])[mask]
    PERTS = _to_np(results["pert_cat"])[mask]
    C_np = _to_np(C).reshape(-1)
    O_np = _to_np(O_pert).reshape(-1) if O_pert is not None else None

    mu_p, mu_t = {}, {}
    uniq = np.unique(PERTS)
    for p in uniq:
        idx = np.where(PERTS == p)[0]
        if idx.size == 0:
            continue
        mu_p[p] = PRED[idx].mean(axis=0)
        mu_t[p] = TRUTH[idx].mean(axis=0)

    pear_list, mse_list, sys_pear_list = [], [], []
    mse_baseline_list = []

    pert2pert_full_id = {}
    geneid2idx = {}

    if adata is not None:
        if "condition" in adata.obs.columns and "condition_name" in adata.obs.columns:
            for cond, cond_name in adata.obs[["condition", "condition_name"]].values:
                pert2pert_full_id[cond] = cond_name

        for i, gene_id in enumerate(adata.var.index.values):
            geneid2idx[gene_id] = i

    for p in mu_p.keys():
        de_idx = None

        if adata is not None and geneid2idx and hasattr(adata, "uns"):
            pert_full_name = pert2pert_full_id.get(p, p)

            de_gene_names = None
            if "top_non_dropout_de_20" in adata.uns and pert_full_name in adata.uns["top_non_dropout_de_20"]:
                de_gene_names = adata.uns["top_non_dropout_de_20"][pert_full_name]
            elif "top_non_zero_de_20" in adata.uns and pert_full_name in adata.uns["top_non_zero_de_20"]:
                de_gene_names = adata.uns["top_non_zero_de_20"][pert_full_name]
            elif "rank_genes_groups_cov_all" in adata.uns and pert_full_name in adata.uns["rank_genes_groups_cov_all"]:
                de_gene_names = adata.uns["rank_genes_groups_cov_all"][pert_full_name][:20]

            if de_gene_names is not None:
                if "gene_name" in adata.var.columns:
                    gene_name_to_id = dict(zip(adata.var["gene_name"], adata.var.index))
                    de_idx = []
                    for gene_name in de_gene_names:
                        if gene_name in gene_name_to_id and gene_name_to_id[gene_name] in geneid2idx:
                            de_idx.append(geneid2idx[gene_name_to_id[gene_name]])
                        elif gene_name in geneid2idx:
                            de_idx.append(geneid2idx[gene_name])
                else:
                    de_idx = [geneid2idx[gene] for gene in de_gene_names if gene in geneid2idx]

                de_idx = de_idx[:20]

        if de_idx is None or len(de_idx) == 0:
            delta_true_abs = np.abs(_to_np(mu_t[p]) - C_np)
            k = min(20, delta_true_abs.shape[0])
            de_idx = np.argsort(-delta_true_abs)[:k]

        if de_idx is None or len(de_idx) == 0:
            continue

        a = _to_np(mu_p[p])[de_idx] - C_np[de_idx]
        b = _to_np(mu_t[p])[de_idx] - C_np[de_idx]
        pear = _pearson(a, b)
        mse = _mse(a, b)
        if np.isfinite(pear):
            pear_list.append(pear)
        if np.isfinite(mse):
            mse_list.append(mse)

        dt = _to_np(mu_t[p])[de_idx] - C_np[de_idx]
        mse_base = np.mean(dt * dt) if dt.size > 0 else np.nan
        if np.isfinite(mse_base):
            mse_baseline_list.append(float(mse_base))

        if O_np is not None:
            a_sys = _to_np(mu_p[p])[de_idx] - O_np[de_idx]
            b_sys = _to_np(mu_t[p])[de_idx] - O_np[de_idx]
            r_sys = _pearson(a_sys, b_sys)
            if np.isfinite(r_sys):
                sys_pear_list.append(r_sys)

    mse_de = float(np.mean(mse_list)) if len(mse_list) else np.nan
    pear_de = float(np.mean(pear_list)) if len(pear_list) else np.nan
    sys_pear_de = float(np.mean(sys_pear_list)) if len(sys_pear_list) else np.nan
    mse_base_avg = float(np.mean(mse_baseline_list)) if len(mse_baseline_list) else np.nan

    if np.isfinite(mse_de) and np.isfinite(mse_base_avg) and mse_base_avg > 0:
        mse_de_norm = float(mse_de / mse_base_avg)
    else:
        mse_de_norm = np.nan

    return {
        "mse_de": mse_de,
        "mse_de_norm": mse_de_norm,
        "pearson_de": pear_de,
        "systema_pearson_de": sys_pear_de,
    }


def _pretty_print_block(name: str, m: dict):
    if name == "overall":
        print(
            f"[Overall] mse={fmt6(m.get('mse'))}  mse_norm={fmt6(m.get('mse_norm'))}  "
            f"pearson={fmt6(m.get('pearson'))}  systema_pearson={fmt6(m.get('systema_pearson'))}"
        )
    elif name == "combo_seen2":
        print(
            f"[Double pert: both genes seen] mse={fmt6(m.get('mse'))}  mse_norm={fmt6(m.get('mse_norm'))}  "
            f"pearson={fmt6(m.get('pearson'))}  systema_pearson={fmt6(m.get('systema_pearson'))}"
        )
    elif name == "combo_seen1":
        print(
            f"[Double pert: one gene seen] mse={fmt6(m.get('mse'))}  mse_norm={fmt6(m.get('mse_norm'))}  "
            f"pearson={fmt6(m.get('pearson'))}  systema_pearson={fmt6(m.get('systema_pearson'))}"
        )
    elif name == "combo_seen0":
        print(
            f"[Double pert: neither gene seen] mse={fmt6(m.get('mse'))}  mse_norm={fmt6(m.get('mse_norm'))}  "
            f"pearson={fmt6(m.get('pearson'))}  systema_pearson={fmt6(m.get('systema_pearson'))}"
        )
    elif name == "unseen_single":
        print(
            f"[Single-gene unseen] mse={fmt6(m.get('mse'))}  mse_norm={fmt6(m.get('mse_norm'))}  "
            f"pearson={fmt6(m.get('pearson'))}  systema_pearson={fmt6(m.get('systema_pearson'))}"
        )
    elif name == "unseen_single_top20":
        print(
            f"[Single-gene unseen, top-20 DE] mse_de={fmt6(m.get('mse_de'))}  "
            f"mse_de_norm={fmt6(m.get('mse_de_norm'))}  pearson_de={fmt6(m.get('pearson_de'))}  "
            f"systema_pearson_de={fmt6(m.get('systema_pearson_de'))}"
        )
    elif name == "all_top20":
        print(
            f"[All perturbations, top-20 DE] mse_de={fmt6(m.get('mse_de'))}  "
            f"mse_de_norm={fmt6(m.get('mse_de_norm'))}  pearson_de={fmt6(m.get('pearson_de'))}  "
            f"systema_pearson_de={fmt6(m.get('systema_pearson_de'))}"
        )
    elif name == "combo_seen2_top20":
        print(
            f"[Double pert both seen, top-20 DE] mse_de={fmt6(m.get('mse_de'))}  "
            f"mse_de_norm={fmt6(m.get('mse_de_norm'))}  pearson_de={fmt6(m.get('pearson_de'))}  "
            f"systema_pearson_de={fmt6(m.get('systema_pearson_de'))}"
        )
    elif name == "combo_seen1_top20":
        print(
            f"[Double pert one seen, top-20 DE] mse_de={fmt6(m.get('mse_de'))}  "
            f"mse_de_norm={fmt6(m.get('mse_de_norm'))}  pearson_de={fmt6(m.get('pearson_de'))}  "
            f"systema_pearson_de={fmt6(m.get('systema_pearson_de'))}"
        )
    elif name == "combo_seen0_top20":
        print(
            f"[Double pert neither seen, top-20 DE] mse_de={fmt6(m.get('mse_de'))}  "
            f"mse_de_norm={fmt6(m.get('mse_de_norm'))}  pearson_de={fmt6(m.get('pearson_de'))}  "
            f"systema_pearson_de={fmt6(m.get('systema_pearson_de'))}"
        )
    else:
        print(
            f"{name}: mse={fmt6(m.get('mse'))}  mse_norm={fmt6(m.get('mse_norm'))}  "
            f"pearson={fmt6(m.get('pearson'))}  systema_pearson={fmt6(m.get('systema_pearson'))}"
        )


def _normalize_subgroup(inner):
    """Normalize subgroup payload from GEARS-style ``test_subgroup`` dict or attribute object."""
    try:
        if inner is None:
            return {}

        if isinstance(inner, dict):
            if "test_subgroup" in inner:
                test_subgroup = inner["test_subgroup"]
            else:
                test_subgroup = inner
        else:
            test_subgroup = inner

        out = {}
        possible_keys = {
            "combo_seen2": ["combo_seen2", "seen2", "double_seen2"],
            "combo_seen1": ["combo_seen1", "seen1", "double_seen1"],
            "combo_seen0": ["combo_seen0", "seen0", "double_seen0"],
            "unseen_single": ["unseen_single", "single_unseen", "single_ood"],
        }

        for target_key, variants in possible_keys.items():
            for variant in variants:
                if hasattr(test_subgroup, variant):
                    perts = getattr(test_subgroup, variant)
                    if perts is not None:
                        out[target_key] = list(perts)
                        break
                elif isinstance(test_subgroup, dict) and variant in test_subgroup:
                    perts = test_subgroup[variant]
                    if perts is not None:
                        out[target_key] = list(perts)
                        break

        return out
    except Exception:
        return {}


def run_assessment(
    model=None,
    loaders=None,
    adata=None,
    device="cpu",
    uncertainty=False,
    C=None,
    O_pert=None,
    subgroup=None,
    results=None,
):
    """Print overall metrics, optional GEARS-style subgroups, and top-20 DE metrics.

    All MSE values include ``mse_norm`` (or ``mse_de_norm``): ratio to the no-change
    baseline that always predicts the control centroid.

    If ``results`` is None, runs :func:`evaluate` on ``loaders['test_loader']``.
    """
    assert C is not None, "run_assessment requires control centroid C."
    if results is None:
        assert model is not None and loaders is not None and "test_loader" in loaders, (
            "When results is None, provide model and loaders['test_loader']."
        )
        results = evaluate(
            loader=loaders["test_loader"],
            model=model,
            uncertainty=uncertainty,
            device=device,
        )

    overall = _overall_metrics(results, C=C, O_pert=O_pert)
    _pretty_print_block("overall", overall)

    S = _normalize_subgroup(subgroup)
    for key in ["combo_seen2", "combo_seen1", "combo_seen0", "unseen_single"]:
        if key not in S:
            S[key] = []
        m = _subset_metrics(results, C=C, O_pert=O_pert, subgroup_list=S[key])
        _pretty_print_block(key, m)

        if key == "unseen_single":
            t20 = _subset_top20(results, C=C, O_pert=O_pert, subgroup_list=S[key], adata=adata)
            _pretty_print_block("unseen_single_top20", t20)

    t20_all = _subset_top20(results, C=C, O_pert=O_pert, subgroup_list=None, adata=adata)
    _pretty_print_block("all_top20", t20_all)

    for key in ["combo_seen2", "combo_seen1", "combo_seen0"]:
        if key in S:
            t20 = _subset_top20(results, C=C, O_pert=O_pert, subgroup_list=S[key], adata=adata)
            _pretty_print_block(f"{key}_top20", t20)

    out = {"overall": overall}
    for key in ["combo_seen2", "combo_seen1", "combo_seen0", "unseen_single"]:
        if key in S:
            out[key] = _subset_metrics(results, C=C, O_pert=O_pert, subgroup_list=S[key])
    if "unseen_single" in S:
        out["unseen_single_top20"] = _subset_top20(
            results, C=C, O_pert=O_pert, subgroup_list=S["unseen_single"], adata=adata
        )
    out["all_top20"] = t20_all
    for key in ["combo_seen2", "combo_seen1", "combo_seen0"]:
        if key in S:
            out[f"{key}_top20"] = _subset_top20(
                results, C=C, O_pert=O_pert, subgroup_list=S[key], adata=adata
            )
    return out


def _compute_O_pert(adata, train_perts=None):
    """Systema reference vector: mean of per-perturbation centroids (non-control).

    Prefers cells with ``obs['split'] == 'train'`` when available; otherwise uses
    ``train_perts`` or all non-control conditions. See Systema documentation:
    https://github.com/mlbio-epfl/systema
    """
    cond = np.asarray(adata.obs["condition"].values)
    X = adata.X

    train_mask = None
    if "split" in adata.obs.columns:
        split_col = adata.obs["split"].astype(str).str.lower()
        train_mask = (split_col == "train").values
    elif train_perts is not None:
        train_perts_set = set(train_perts)
        train_mask = np.array([c in train_perts_set for c in cond])

    if train_perts is None:
        perts = np.unique(cond)
        perts = [p for p in perts if p != "ctrl"]
    else:
        perts = [p for p in list(train_perts) if p != "ctrl"]

    centroids = []
    for p in perts:
        idx = np.where(cond == p)[0]
        if idx.size == 0:
            continue
        if train_mask is not None:
            idx = idx[train_mask[idx]]
        if idx.size == 0:
            continue
        mu = np.asarray(X[idx].mean(axis=0)).reshape(-1)
        centroids.append(mu)

    if len(centroids) == 0:
        if train_mask is not None:
            ctrl_idx = np.where((cond == "ctrl") & train_mask)[0]
        else:
            ctrl_idx = np.where(cond == "ctrl")[0]
        if ctrl_idx.size == 0:
            return np.asarray(X.mean(axis=0)).reshape(-1)
        return np.asarray(X[ctrl_idx].mean(axis=0)).reshape(-1)

    O_pert = np.stack(centroids, 0).mean(axis=0)
    return O_pert.reshape(-1)
