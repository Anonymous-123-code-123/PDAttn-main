from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

from data_tools.PertData import PertData
from tools.assessment import _compute_O_pert, run_assessment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the gene-specific linear model (GSLM) baseline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiankampmann2021_crispri",
        help=(
            "Dataset name passed to PertData.load, e.g. norman, dixit, adamson, "
            "xu_kinetics_2024, replogle_k562_essential, replogle_rpe1_essential, "
            "tiankampmann2021_crispra, tiankampmann2021_crispri."
        ),
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory for PertData.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prepare_split.")
    parser.add_argument("--split", type=str, default="simulation", help="Split mode (see PertData.prepare_split).")
    parser.add_argument(
        "--train_gene_set_size",
        type=float,
        default=0.75,
        help="Fraction of perturbation genes in the training set (simulation splits).",
    )
    parser.add_argument("--pca_dim", type=int, default=64, help="PCA latent dimension (capped by n_samples / n_features).")
    parser.add_argument("--alpha", type=float, default=0.1, help="Ridge L2 penalty (per-gene regressors).")

    return parser.parse_args()


def extract_paired_data_from_dataloader(loader):
    """Stack PyG batches into ``X`` (control expression) and ``y`` (perturbed), plus condition strings."""
    X_list = []
    y_list = []
    cond_list = []

    for batch in loader:
        num_nodes = batch.x.shape[0]
        num_genes = num_nodes // batch.num_graphs
        x_reshaped = batch.x.view(batch.num_graphs, num_genes)
        y_reshaped = batch.y.view(batch.num_graphs, num_genes)
        X_list.append(x_reshaped.numpy())
        y_list.append(y_reshaped.numpy())
        cond_list.extend(batch.pert)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    conditions = cond_list

    return X, y, conditions


class GeneSpecificLinearModel:
    """Per-gene Ridge on PCA features of input expression."""

    def __init__(self, pca_dim: int = 64, alpha: float = 1.0) -> None:
        self.pca_dim = pca_dim
        self.alpha = alpha
        self.pca: PCA | None = None
        self.models: list | None = None
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.n_genes: int | None = None
        self.C_ctrl: np.ndarray | None = None
        self.gene_names: list | None = None

    def fit(self, X_train, y_train, C_ctrl, gene_names=None) -> None:
        self.C_ctrl = C_ctrl
        self.n_genes = X_train.shape[1]
        self.gene_names = gene_names

        print(f"Training arrays: X_train={X_train.shape}, y_train={y_train.shape}")

        actual_pca_dim = min(self.pca_dim, X_train.shape[1], X_train.shape[0])
        print(f"PCA: {X_train.shape[1]} features -> {actual_pca_dim} components")

        self.pca = PCA(n_components=actual_pca_dim)
        X_train_pca = self.pca.fit_transform(X_train)

        self.models = []
        self.W = np.zeros((self.n_genes, actual_pca_dim))
        self.b = np.zeros(self.n_genes)

        print("Fitting per-gene Ridge models...")
        for g in range(self.n_genes):
            if g % 100 == 0:
                print(f"  Progress: {g}/{self.n_genes} genes")

            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X_train_pca, y_train[:, g])
            self.models.append(model)
            self.W[g] = model.coef_
            self.b[g] = model.intercept_

        print(f"  Done: {self.n_genes}/{self.n_genes} genes")

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.pca is not None and self.W is not None and self.b is not None
        X_pca = self.pca.transform(X)
        return X_pca @ self.W.T + self.b


def create_results_dict_for_assessment(y_pred, y_true, conditions) -> dict:
    """Shape outputs like neural ``evaluate`` so ``run_assessment`` can consume them."""
    return {
        "pred": y_pred,
        "truth": y_true,
        "pert_cat": conditions,
        "pred_de": [],
        "truth_de": [],
        "de_idx": [],
        "logvar": None,
    }


def main():
    args = parse_args()

    print(f"\n{'=' * 80}")
    print(f"{'Gene-specific linear model (GSLM)':^80}")
    print(f"{('Dataset: ' + args.dataset):^80}")
    print(f"{'=' * 80}\n")

    print("Step 1: load PertData...")
    try:
        pert = PertData(args.data_dir)
        pert.load(data_name=args.dataset)
        pert.prepare_split(
            split=args.split,
            seed=args.seed,
            train_gene_set_size=args.train_gene_set_size,
        )
        print(f"Loaded: {args.dataset}, AnnData shape {pert.adata.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nStep 2: dataloaders...")
    pert.get_dataloader(batch_size=128, test_batch_size=128)

    train_loader = pert.dataloader["train_loader"]
    test_loader = pert.dataloader["test_loader"]
    print(f"Train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    print("\nStep 3: materialize (X, y) from PyG loaders...")
    X_train, y_train, _cond_train = extract_paired_data_from_dataloader(train_loader)
    X_test, y_test, cond_test = extract_paired_data_from_dataloader(test_loader)
    print(f"X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"X_test {X_test.shape}, y_test {y_test.shape}")

    print("\nStep 4: global baseline vector and overlap matrix...")
    C_ctrl = np.mean(X_train, axis=0)
    print(f"C_ctrl shape {C_ctrl.shape}")

    if hasattr(pert, "set2conditions") and pert.set2conditions and "train" in pert.set2conditions:
        train_perts = pert.set2conditions["train"]
    else:
        train_perts = None

    O_pert = _compute_O_pert(pert.adata, train_perts=train_perts)
    print(f"O_pert shape: {O_pert.shape if O_pert is not None else None}")

    if "gene_name" in pert.adata.var.columns:
        gene_names = list(pert.adata.var["gene_name"].values)
    else:
        gene_names = list(pert.adata.var.index.values)
    print(f"n_genes (names): {len(gene_names)}")

    print("\nStep 5: train GSLM...")
    model = GeneSpecificLinearModel(pca_dim=args.pca_dim, alpha=args.alpha)
    t0 = time.time()
    model.fit(X_train, y_train, C_ctrl, gene_names)
    print(f"Training time: {time.time() - t0:.2f} s")

    print("\nStep 6: predict on test...")
    y_pred = model.predict(X_test)
    print(f"y_pred {y_pred.shape}, y_true {y_test.shape}")

    print("\nStep 7: run_assessment...")
    results = create_results_dict_for_assessment(y_pred, y_test, cond_test)

    class _StubTorchModel:
        """Placeholder; metrics are computed from ``results`` dict, not forward passes."""

        def eval(self):
            return self

    dummy_model = _StubTorchModel()
    dummy_loaders = {"test_loader": None}
    subgroup = getattr(pert, "subgroup", None)

    C_ctrl_tensor = torch.from_numpy(C_ctrl).float()
    O_pert_tensor = torch.from_numpy(O_pert).float() if O_pert is not None else None

    print("\n" + "=" * 80)
    print(f"{'Metrics':^80}")
    print("=" * 80)

    metrics = run_assessment(
        model=dummy_model,
        loaders=dummy_loaders,
        adata=pert.adata,
        device="cpu",
        uncertainty=False,
        C=C_ctrl_tensor,
        O_pert=O_pert_tensor,
        subgroup=subgroup,
        results=results,
    )

    print("\n" + "=" * 80)
    print(f"{'Finished.':^80}")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    main()
