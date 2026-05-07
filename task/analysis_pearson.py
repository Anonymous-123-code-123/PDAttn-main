"""PDAttn **inference and evaluation** entry point (CLI).

Loads a trained checkpoint, runs the model on the **test** split, and prints metrics via
``tools.assessment.run_assessment`` (overall / GEARS-style subgroups / top-20 DE, including
Systema-Pearson when ``O_pert`` is available). Use the same ``--dataset``, ``--split``, and
``--seed`` as training so splits and references match the saved run.

Typical usage::

    python -m task.analysis_pearson --dataset norman --ckpt model/norman.pt --gpu 0

Run from the repository root (or set ``PYTHONPATH``) so ``PDAttn``, ``tools``, and
``data_tools`` packages resolve.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from data_tools.PertData import PertData
from PDAttn.G_model import GenePertGraphBuilder
from PDAttn.PDATT import build_pdattn_model
from tools import assessment as ASM
from tools.inference import evaluate
from tools.utils import setup_gpu_device, use_data_parallel_if_available


def _pick_test_loader(pert, batch_size: int, test_batch_size: Optional[int] = None):
    loaders = pert.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
    if loaders is None:
        loaders = getattr(pert, "dataloader", None)
    if not isinstance(loaders, dict) or "test_loader" not in loaders:
        keys = list(loaders.keys()) if isinstance(loaders, dict) else "n/a"
        raise RuntimeError(
            f"Missing or invalid dataloaders (type {type(loaders)}); available keys: {keys}"
        )
    return loaders["test_loader"]


@torch.no_grad()
def _ctrl_centroid_from_adata(adata):
    cond = np.asarray(adata.obs["condition"].values)
    X = adata.X
    C = np.asarray(X[(cond == "ctrl")].mean(axis=0)).reshape(-1)
    return torch.as_tensor(C, dtype=torch.float32)


def _resolve_struct_args(saved_args: dict, argsNS) -> dict:
    """Restore structural hyperparameters from checkpoint ``args``; CLI ``override_*`` wins.

    Expected training keys: ``hidden_dim``, ``dropout``, ``num_attention_heads``.
    """
    def pick(key, default):
        val = getattr(argsNS, f"override_{key}", None)
        if val is not None:
            return val
        return saved_args.get(key, default)

    hidden_dim = int(pick("hidden_dim", 64))
    dropout = float(pick("dropout", 0.0))
    num_attention_heads = int(pick("num_attention_heads", 4))
    return dict(
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_attention_heads=num_attention_heads,
    )


def main():
    parser = argparse.ArgumentParser(
        description="PDAttn test-set evaluation (overall, subgroups, top-20 DE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="norman", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data root directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id(s), e.g. '0' or '0,1'; use 'cpu' for CPU",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="simulation",
        help="Split name (must match training)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training)")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (.pt)")
    parser.add_argument("--override_hidden_dim", type=int, default=None, help="Override hidden size")
    parser.add_argument("--override_dropout", type=float, default=None, help="Override dropout")
    parser.add_argument(
        "--override_num_attention_heads",
        type=int,
        default=None,
        help="Override number of attention heads",
    )

    args = parser.parse_args()

    device = setup_gpu_device(str(args.gpu))

    pert = PertData(args.data_dir)
    pert.load(data_name=args.dataset)
    pert.prepare_split(split=args.split, seed=args.seed, train_gene_set_size=0.75)

    print("Building test DataLoader...")
    test_loader = _pick_test_loader(pert, batch_size=args.batch_size, test_batch_size=args.batch_size)
    print("DataLoader ready.")

    builder = GenePertGraphBuilder(
        adata=pert.adata,
        gene_list=list(pert.adata.var["gene_name"].values),
        pert_list=list(getattr(pert, "pert_names", [])),
        node_map=getattr(
            pert,
            "node_map",
            {g: i for i, g in enumerate(pert.adata.var["gene_name"].values)},
        ),
        node_map_pert=getattr(pert, "node_map_pert", {}),
        data_path=pert.data_path,
        dataset_name=pert.dataset_name,
        split=pert.split,
        seed=getattr(pert, "seed", args.seed),
        train_gene_set_size=getattr(pert, "train_gene_set_size", 0.75),
        set2conditions=getattr(pert, "set2conditions", {}),
        default_pert_graph=getattr(pert, "default_pert_graph", True),
        k_go=20,
        coexpr_threshold=0.4,
        device=device,
    )

    ckpt_path = args.ckpt or os.path.join("model", f"{args.dataset}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {}) or {}

    builder.build_all()

    struct = _resolve_struct_args(saved_args, args)
    model = build_pdattn_model(
        builder,
        hidden_dim=struct["hidden_dim"],
        num_attention_heads=struct["num_attention_heads"],
        dropout=struct["dropout"],
    ).to(device)
    model = use_data_parallel_if_available(model)

    state = ckpt.get("model_state", None)
    if state is None:
        raise KeyError("Checkpoint missing key 'model_state'.")

    real_model = model.module if hasattr(model, "module") else model
    real_model.load_state_dict(state, strict=True)
    model.eval()

    results = evaluate(loader=test_loader, model=model, uncertainty=False, device=device)

    C_ctrl_ckpt = ckpt.get("C_ctrl", None)
    C_ctrl = (
        C_ctrl_ckpt.to(device)
        if C_ctrl_ckpt is not None
        else _ctrl_centroid_from_adata(pert.adata).to(device)
    )
    O_pert = ckpt.get("mu_ref", None)

    _ = ASM.run_assessment(
        model=None,
        loaders=None,
        adata=pert.adata,
        device=device,
        uncertainty=False,
        C=C_ctrl,
        O_pert=O_pert,
        subgroup=getattr(pert, "subgroup", None),
        results=results,
    )


if __name__ == "__main__":
    main()