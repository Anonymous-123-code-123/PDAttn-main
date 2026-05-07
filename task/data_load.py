"""Bulk data preparation for **PDAttn** experiments (GEARS-style ``PertData``).

Iterates a fixed list of perturbation-seq datasets, loads or ingests each into
``data_tools.PertData`` under ``./data``, then runs ``prepare_split`` (default
``simulation``) and ``get_dataloader`` so artifacts match the training pipeline
used elsewhere in this repo (API aligned with `GEARS
<https://github.com/snap-stanford/GEARS>`_).

**Built-in downloads** (via ``PertData.load``): e.g. Norman, Adamson, Replogle
essentials. **Manual assets** (place under ``./data`` before running):

- Tian CRISPRa / CRISPRi: Zenodo
  https://zenodo.org/records/13350497 (``.h5ad`` files).
- Xu kinetics: GEO `GSE218566` (e.g. ``GSM6752591_*`` matrix + metadata as used
  below).

Run from the repository root::

    python -m task.data_load
"""
from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from data_tools.PertData import PertData


def load_data() -> None:
    """Download / preprocess listed datasets and build train–test loaders (seed fixed)."""
    datasets = [
        "norman",
        "dixit"
        "adamson",
        "replogle_k562_essential",
        "replogle_rpe1_essential",
        "TianKampmann2021_CRISPRa",
        "xu_kinetics_2024",
        "TianKampmann2021_CRISPRi",
    ]

    for data_name in datasets:
        print(f"******************** Loading {data_name} ********************")
        pert_data = PertData("./data")

        if data_name == "TianKampmann2021_CRISPRa":
            data_dir = "./data"
            # Optional raw→GEARS format: build ``adata_subset``, then
            # ``PertData.new_data_process("tiankampmann2021_crispra", adata_subset)``
            # before ``load`` if ``perturb_processed.h5ad`` is not yet on disk.
            adata = anndata.read_h5ad(f"{data_dir}/TianKampmann2021_CRISPRa.h5ad")
            # Harmonize symbol aliases with reference gene names
            target_genes = adata.obs["perturbation"].tolist()
            target_genes[target_genes == "ATP5C1"] = "ATP5F1C"
            target_genes[target_genes == "SMEK1"] = "PPP4R3A"
            adata.obs["perturbation"] = target_genes
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)
            adata.var["gene_name"] = adata.var.index
            adata.obs["gene"] = adata.obs["perturbation"]
            hvg_flag_ = adata.var["highly_variable"].values
            gene_flag_ = adata.var["gene_name"].isin(adata.obs["gene"].values).values
            select_flag_ = np.logical_or(hvg_flag_, gene_flag_)
            condition_flag_ = adata.obs["gene"].isin(
                adata.var["gene_name"].values.tolist() + ["control"]
            ).values
            adata_subset = adata[condition_flag_, select_flag_]
            pert_counts = adata_subset.obs["gene"].value_counts()
            keep_perts = pert_counts[pert_counts > 1].index.values
            adata_subset = adata_subset[adata_subset.obs["gene"].isin(keep_perts)]
            adata_subset.obs["condition"] = [i + "+ctrl" for i in adata_subset.obs["gene"].values]
            adata_subset.obs["condition"] = adata_subset.obs["condition"].replace(
                {"control+ctrl": "ctrl"}
            )
            adata_subset.obs["cell_type"] = "iPSC"
            pert_data.load(data_name="tiankampmann2021_crispra")

        elif data_name == "xu_kinetics_2024":
            data_dir = "./data"
            # Same pattern: raw mtx + metadata → ``adata_subset``; persist via
            # ``new_data_process`` when starting from GEO files only.
            adata_ = anndata.read_mtx(f"{data_dir}/GSM6752591_on_target_whole_tx_count_matrix.mtx.gz")
            obs = pd.read_csv(
                f"{data_dir}/GSM6752591_on_target_cell_metadata.csv.gz",
                sep=",",
                compression="gzip",
            )
            var = pd.read_csv(
                f"{data_dir}/GSM6752591_on_target_whole_tx.Genes.tsv.gz",
                sep=",",
                compression="gzip",
                header=None,
            )
            var.columns = ["gene_name"]
            adata = anndata.AnnData(adata_.X.T, obs=obs, var=var)
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)
            adata.obs["gene"] = adata.obs["target_genes"]
            hvg_flag_ = adata.var["highly_variable"].values
            gene_flag_ = adata.var["gene_name"].isin(adata.obs["gene"].values).values
            select_flag_ = np.logical_or(hvg_flag_, gene_flag_)
            condition_flag_ = adata.obs["gene"].isin(
                adata.var["gene_name"].values.tolist() + ["NO-TARGET"]
            ).values
            adata_subset = adata[condition_flag_, select_flag_]
            adata_subset.obs["condition"] = [i + "+ctrl" for i in adata_subset.obs["gene"].values]
            adata_subset.obs["condition"] = adata_subset.obs["condition"].replace(
                {"NO-TARGET+ctrl": "ctrl"}
            )
            adata_subset.obs["cell_type"] = "HEK293"
            pert_data.load(data_name="xu_kinetics_2024")

        elif data_name == "TianKampmann2021_CRISPRi":
            data_dir = "./data"
            # Optional preprocessing recipe (CRISPRi); ship ``perturb_processed.h5ad``
            # under ``./data/tiankampmann2021_crispri/`` for ``load`` to succeed.
            adata = anndata.read_h5ad(f"{data_dir}/TianKampmann2021_CRISPRi.h5ad")
            target_genes = adata.obs["perturbation"].tolist()
            target_genes[target_genes == "TMEM55A"] = "PIP4P2"
            target_genes[target_genes == "ATP5C1"] = "ATP5F1C"
            target_genes[target_genes == "ATP5H"] = "ATP5PD"
            adata.obs["perturbation"] = target_genes
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)
            adata.var["gene_name"] = adata.var.index
            adata.obs["gene"] = adata.obs["perturbation"]
            hvg_flag_ = adata.var["highly_variable"].values
            gene_flag_ = adata.var["gene_name"].isin(adata.obs["gene"].values).values
            select_flag_ = np.logical_or(hvg_flag_, gene_flag_)
            condition_flag_ = adata.obs["gene"].isin(
                adata.var["gene_name"].values.tolist() + ["control"]
            ).values
            adata_subset = adata[condition_flag_, select_flag_]
            pert_counts = adata_subset.obs["gene"].value_counts()
            keep_perts = pert_counts[pert_counts > 1].index.values
            adata_subset = adata_subset[adata_subset.obs["gene"].isin(keep_perts)]
            adata_subset.obs["condition"] = [i + "+ctrl" for i in adata_subset.obs["gene"].values]
            adata_subset.obs["condition"] = adata_subset.obs["condition"].replace(
                {"control+ctrl": "ctrl"}
            )
            adata_subset.obs["cell_type"] = "iPSC"
            pert_data.load(data_name="tiankampmann2021_crispri")
        else:
            pert_data.load(data_name=data_name)

        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=32, test_batch_size=128)
        print(f"******************** Finished {data_name} ********************")
        print("\n")


if __name__ == "__main__":
    print("\n")
    load_data()
