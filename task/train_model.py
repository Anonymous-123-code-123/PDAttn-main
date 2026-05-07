"""PDAttn **training** CLI (single or multi-dataset).

Loads data with ``PertData`` using **GEARS-style** splits and PyG dataloaders
(`GEARS <https://github.com/snap-stanford/GEARS>`_), builds the GO / co-expression
graph via :class:`PDAttn.G_model.GenePertGraphBuilder`, optimizes
:class:`PDAttn.PDATT` with :class:`tools.Loss.SystemaLoss`, and writes checkpoints
under ``model/<dataset_name>.pt`` (state dict, ``C_ctrl``, optional ``mu_ref`` /
``O_pert`` for assessment, args, last validation metrics).

Examples::

    python -m task.train_model --dataset norman --gpu 0 --epochs 15
    python -m task.train_model --dataset norman,adamson --data_dir ./data --seed 42

Use the same ``--data_dir``, ``split`` (fixed ``simulation`` here), and ``--seed``
as evaluation scripts so splits match.
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

import torch
from torch.optim import AdamW

from data_tools.PertData import PertData
from PDAttn.G_model import GenePertGraphBuilder
from PDAttn.PDATT import build_pdattn_model, estimate_ctrl_and_pert_centroids
from PDAttn.Train import train_one_epoch, validate_one_epoch
from tools.Loss import SystemaLoss
from tools.assessment import _compute_O_pert
from tools.utils import (
    seed_all,
    setup_gpu_device,
    use_data_parallel_if_available,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PDAttn on one or more datasets (comma-separated --dataset)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="norman",
        help=(
            "PertData.load name(s), comma-separated, e.g. "
            "norman,adamson,replogle_k562_essential,tiankampmann2021_crispri,..."
        ),
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory passed to PertData.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Train DataLoader batch size.")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Val/test DataLoader batch size.")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Device id(s), e.g. '0' or '0,1'; use 'cpu' for CPU.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed (split + training).")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    args = parser.parse_args()

    datasets = [ds.strip() for ds in args.dataset.split(",") if ds.strip()]
    if not datasets:
        print("Error: no valid dataset names in --dataset.")
        return

    print(f"\nTraining on {len(datasets)} dataset(s): {datasets}\n")

    seed_all(args.seed)
    device = setup_gpu_device(str(args.gpu))

    total_start_time = time.time()
    for idx, dataset_name in enumerate(datasets, 1):
        try:
            print(f"\nProgress: [{idx}/{len(datasets)}]")
            train_single_dataset(dataset_name, args, device)
        except Exception as e:
            print(f"\nError while training '{dataset_name}': {e}")
            print("Continuing with the next dataset...\n")
            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time
    print(f"\n{'=' * 80}")
    print("All scheduled runs finished.")
    print(f"Datasets in this job: {len(datasets)}")
    print(f"Total wall time: {total_time / 3600:.2f} h ({total_time:.2f} s)")
    print(f"{'=' * 80}\n")


def train_single_dataset(dataset_name: str, args: argparse.Namespace, device: torch.device) -> None:
    """Load one benchmark, build graph + model, fit, and save ``model/{name}.pt`` each epoch."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}\n")

    pert = PertData(args.data_dir)
    pert.load(data_name=dataset_name)
    pert.prepare_split(split="simulation", seed=args.seed, train_gene_set_size=0.75)
    pert.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)
    loaders = {
        "train_loader": pert.dataloader["train_loader"],
        "val_loader": pert.dataloader["val_loader"],
    }

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

    t0 = time.time()
    print(f"[{dataset_name}] Building perturbation graphs...")
    _ = builder.build_all()
    t1 = time.time()
    print(f"[{dataset_name}] Graph build done in {t1 - t0:.2f} s")

    model = build_pdattn_model(builder).to(device)
    t2 = time.time()
    print(f"[{dataset_name}] Model init in {t2 - t1:.2f} s")

    model = use_data_parallel_if_available(model)
    t3 = time.time()
    print(f"[{dataset_name}] DataParallel (if any) in {t3 - t2:.2f} s")

    train_size = len(loaders["train_loader"])
    max_batches_for_estimation = train_size
    t4 = time.time()

    # Control centroid used in loss and validation (delta vs ctrl).
    C_ctrl, _ = estimate_ctrl_and_pert_centroids(
        loaders["train_loader"],
        device=device,
        ctrl_name="ctrl",
        max_batches=max_batches_for_estimation,
        verbose=True,
    )

    t5 = time.time()
    print(f"[{dataset_name}] C_ctrl estimated in {t5 - t4:.2f} s")

    loss_fn = SystemaLoss().to(device)
    loss_fn.set_systema_refs(mu_ref=None, delta_sys_mean=None)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    os.makedirs("model", exist_ok=True)
    save_path = f"model/{dataset_name}.pt"

    print(f"[{dataset_name}] Training for {args.epochs} epoch(s)...")

    for ep in range(1, args.epochs + 1):
        _ = train_one_epoch(
            model=model,
            train_loader=loaders["train_loader"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            C_ctrl=C_ctrl,
            device=device,
            epoch=ep,
            print_every=50,
        )

        metrics = validate_one_epoch(
            model=model,
            val_loader=loaders["val_loader"],
            device=device,
            epoch=ep,
            C_ctrl=C_ctrl,
            pert=pert,
        )

        mu_ref_saved = None
        if pert is not None:
            try:
                train_perts = None
                if hasattr(pert, "set2conditions") and pert.set2conditions and "train" in pert.set2conditions:
                    train_perts = pert.set2conditions["train"]
                mu_ref_saved = _compute_O_pert(pert.adata, train_perts=train_perts)
                mu_ref_saved = torch.from_numpy(mu_ref_saved).float().cpu()
            except Exception:
                mu_ref_saved = None

        real = model.module if hasattr(model, "module") else model
        torch.save(
            {
                "model_state": real.state_dict(),
                "C_ctrl": C_ctrl.cpu(),
                "mu_ref": mu_ref_saved,
                "args": vars(args),
                "metrics": metrics,
            },
            save_path,
        )
        print(f"[{dataset_name}] Epoch {ep}/{args.epochs} - saved: {save_path}")

    print(f"\n[{dataset_name}] Finished.\n")


if __name__ == "__main__":
    main()
