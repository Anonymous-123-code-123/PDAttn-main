"""Perturbation dataset loading, preprocessing, and train/val/test splits for PDAttn.

This module follows the **GEARS** data API and splitting conventions so results are
comparable to the standard single-cell perturbation benchmark. Reference:

  https://github.com/snap-stanford/GEARS

Split modes (see :meth:`PertData.prepare_split`) mirror GEARS ``PertData.prepare_split``:

- **simulation** — Main benchmark split: hold out genes for unseen single-gene
  perturbations and stratify double perturbations into difficulty buckets.
  Produces ``subgroup`` / ``test_subgroup`` with:

  - **unseen_single** — single-gene perturbations whose target gene was not in the
    training gene set.
  - **combo_seen2** — double perturbations where both genes appear in training, but
    the specific combination was held out for test.
  - **combo_seen1** — double perturbation with exactly one gene in the training set.
  - **combo_seen0** — double perturbation with neither gene in the training set.

- **simulation_single** — Same gene-holdout idea restricted to **single-gene**
  perturbations only (no double-pert training mix in the same way as full simulation).

- **combo_seen0**, **combo_seen1**, **combo_seen2** — Splits focused on double
  perturbations where **0 / 1 / 2** genes of the pair were seen during training
  (``seen`` passed to :class:`data_utils.DataSplitter`).

- **single** — Random or gene-targeted holdout of **single-gene** perturbations
  (fraction controlled by ``combo_single_split_test_set_fraction``).

- **no_test** — Only **train** and **val** (no held-out test split).

- **no_split** — All cells labeled **test** (e.g. inference-only or external data).

- **custom** — Load a pickled ``set2conditions`` dict from ``split_dict_path``
  (train / val / test condition lists).

PyG **cell graphs** are stored in ``data_pyg/cell_graphs.pkl``; use
``load(..., load_cell_graphs=False)`` when only ``AnnData`` / splits are needed.
"""
from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
import scanpy as sc
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from data_tools.data_utils import (
    get_DE_genes,
    get_dropout_non_zero_genes,
    DataSplitter,
    print_sys,
    zip_data_download_wrapper,
    dataverse_download,
    filter_pert_in_go,
    get_genes_from_perts,
)


class PertData:
    """Load perturbation scRNA-seq data, optional PyG cell graphs, and GEARS-style splits."""

    def __init__(self, data_path, gene_set_path=None, default_pert_graph=True):
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        server_path = "https://dataverse.harvard.edu/api/access/datafile/6153417"
        dataverse_download(server_path, os.path.join(self.data_path, "gene2go_all.pkl"))
        with open(os.path.join(self.data_path, "gene2go_all.pkl"), "rb") as f:
            self.gene2go = pickle.load(f)

    def set_pert_genes(self):
        """Define the perturbation-gene universe for GO graph construction."""
        if self.gene_set_path is not None:
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, "rb") as f:
                essential_genes = pickle.load(f)

        elif self.default_pert_graph is False:
            all_pert_genes = get_genes_from_perts(self.adata.obs["condition"])
            essential_genes = list(self.adata.var["gene_name"].values)
            essential_genes += all_pert_genes

        else:
            server_path = "https://dataverse.harvard.edu/api/access/datafile/6934320"
            path_ = os.path.join(self.data_path, "essential_all_data_pert_genes.pkl")
            dataverse_download(server_path, path_)
            with open(path_, "rb") as f:
                essential_genes = pickle.load(f)

        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}

    def load(self, data_name=None, data_path=None, load_cell_graphs=True):
        """Load a public or custom processed dataset (``perturb_processed.h5ad``).

        If ``load_cell_graphs`` is False, skip loading ``data_pyg/cell_graphs.pkl`` to
        save memory; set True before training (``get_dataloader`` requires graphs).
        """
        if data_name in [
            "norman",
            "adamson",
            "dixit",
            "replogle_k562_essential",
            "replogle_rpe1_essential",
            "xu_kinetics_2024",
            "tiankampmann2021_crispra",
            "tiankampmann2021_crispri",
        ]:
            if data_name == "norman":
                url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
            elif data_name == "adamson":
                url = "https://dataverse.harvard.edu/api/access/datafile/6154417"
            elif data_name == "dixit":
                url = "https://dataverse.harvard.edu/api/access/datafile/6154416"
            elif data_name == "replogle_k562_essential":
                url = "https://dataverse.harvard.edu/api/access/datafile/7458695"
            elif data_name == "replogle_rpe1_essential":
                url = "https://dataverse.harvard.edu/api/access/datafile/7458694"
            else:
                url = "Download independently through links and instructions"

            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)

            self.dataset_name = data_path.split("/")[-1]
            self.dataset_path = data_path

            adata_path = os.path.join(data_path, "perturb_processed.h5ad")
            self.adata = sc.read_h5ad(adata_path)

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, "perturb_processed.h5ad")
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split("/")[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("Expected a known public data_name or an existing data_path with perturb_processed.h5ad")

        self.set_pert_genes()

        print_sys("Perturbations not in GO graph (will be filtered):")
        not_in_go_pert = np.array(
            self.adata.obs[
                self.adata.obs.condition.apply(lambda x: not filter_pert_in_go(x, self.pert_names))
            ].condition.unique()
        )
        print_sys(not_in_go_pert)

        filter_go_mask = self.adata.obs[
            self.adata.obs.condition.apply(lambda x: filter_pert_in_go(x, self.pert_names))
        ]
        self.adata = self.adata[filter_go_mask.index.values, :]

        pyg_path = os.path.join(data_path, "data_pyg")
        dataset_fname = os.path.join(pyg_path, "cell_graphs.pkl")

        if not load_cell_graphs:
            print_sys(
                "Skipping cell graphs (load_cell_graphs=False); only AnnData loaded. "
                "Use load(..., load_cell_graphs=True) before get_dataloader for training."
            )
            self.dataset_processed = None
        else:
            if not os.path.exists(pyg_path):
                os.mkdir(pyg_path)

            if os.path.isfile(dataset_fname):
                sz_gb = os.path.getsize(dataset_fname) / (1024**3)
                if sz_gb >= 0.5:
                    print_sys(
                        f"Loading cached cell graphs (~{sz_gb:.2f} GB)... "
                        "(Ctrl+C if OOM; use load_cell_graphs=False for analysis-only scripts.)"
                    )
                else:
                    print_sys("Loading cached cell graphs...")
                with open(dataset_fname, "rb") as f:
                    self.dataset_processed = pickle.load(f)
                gc.collect()
                print_sys("Done loading cell graphs.")
            else:
                self.ctrl_adata = self.adata[self.adata.obs["condition"] == "ctrl"]
                self.gene_names = self.adata.var.gene_name

                print_sys("Building per-cell PyG graphs...")
                self.create_dataset_file()
                print_sys(f"Saving cell graphs to {dataset_fname}")
                with open(dataset_fname, "wb") as f:
                    pickle.dump(self.dataset_processed, f)
                print_sys("Finished writing cell graphs.")

    def new_data_process(self, dataset_name, adata=None, skip_calc_de=False):
        """Process raw AnnData into GEARS-style ``perturb_processed.h5ad`` and cell graphs."""
        if "condition" not in adata.obs.columns.values:
            raise ValueError("AnnData.obs must contain 'condition' column.")
        if "gene_name" not in adata.var.columns.values:
            raise ValueError("AnnData.var must contain 'gene_name' column.")
        if "cell_type" not in adata.obs.columns.values:
            raise ValueError("AnnData.obs must contain 'cell_type' column.")

        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)

        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder

        self.adata = get_DE_genes(adata, skip_calc_de)
        if not skip_calc_de:
            self.adata = get_dropout_non_zero_genes(self.adata)
        self.adata.write_h5ad(os.path.join(save_data_folder, "perturb_processed.h5ad"))

        self.set_pert_genes()
        self.ctrl_adata = self.adata[self.adata.obs["condition"] == "ctrl"]
        self.gene_names = self.adata.var.gene_name
        pyg_path = os.path.join(save_data_folder, "data_pyg")
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, "cell_graphs.pkl")
        print_sys("Building per-cell PyG graphs...")
        self.create_dataset_file()
        print_sys(f"Saving cell graphs to {dataset_fname}")
        pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))
        print_sys("Finished writing cell graphs.")

    def prepare_split(
        self,
        split="simulation",
        seed=1,
        train_gene_set_size=0.75,
        combo_seen2_train_frac=0.75,
        combo_single_split_test_set_fraction=0.1,
        test_perts=None,
        only_test_set_perts=False,
        test_pert_genes=None,
        split_dict_path=None,
    ):
        """Assign ``adata.obs['split']`` via GEARS-compatible strategies (see module docstring)."""
        available_splits = [
            "simulation",
            "simulation_single",
            "combo_seen0",
            "combo_seen1",
            "combo_seen2",
            "single",
            "no_test",
            "no_split",
            "custom",
        ]
        if split not in available_splits:
            raise ValueError(f"split must be one of: {','.join(available_splits)}")
        self.split = split
        self.seed = seed
        self.subgroup = None

        if split == "custom":
            try:
                with open(split_dict_path, "rb") as f:
                    self.set2conditions = pickle.load(f)
            except Exception as e:
                raise ValueError("custom split requires valid split_dict_path (pickle of set2conditions).") from e
            return

        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, "splits")
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = f"{self.dataset_name}_{split}_{seed}_{train_gene_set_size}.pkl"
        split_path = os.path.join(split_folder, split_file)

        if test_perts:
            split_path = split_path[:-4] + "_" + test_perts + ".pkl"

        if os.path.exists(split_path):
            print_sys("Loading cached split...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split in ("simulation", "simulation_single"):
                subgroup_path = split_path[:-4] + "_subgroup.pkl"
                if os.path.isfile(subgroup_path):
                    self.subgroup = pickle.load(open(subgroup_path, "rb"))
        else:
            print_sys("Creating new split...")
            if test_perts:
                test_perts = test_perts.split("_")

            if split in ["simulation", "simulation_single"]:
                DS = DataSplitter(self.adata, split_type=split)
                adata, subgroup = DS.split_data(
                    train_gene_set_size=train_gene_set_size,
                    combo_seen2_train_frac=combo_seen2_train_frac,
                    seed=seed,
                    test_perts=test_perts,
                    only_test_set_perts=only_test_set_perts,
                )
                subgroup_path = split_path[:-4] + "_subgroup.pkl"
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup

            elif split[:5] == "combo":
                seen = int(split[-1])
                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split("_")
                DS = DataSplitter(self.adata, split_type="combo", seen=seen)
                adata = DS.split_data(
                    test_size=combo_single_split_test_set_fraction,
                    test_perts=test_perts,
                    test_pert_genes=test_pert_genes,
                    seed=seed,
                )

            elif split == "single":
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(
                    test_size=combo_single_split_test_set_fraction,
                    seed=seed,
                )

            elif split == "no_test":
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)

            elif split == "no_split":
                adata = self.adata
                adata.obs["split"] = "test"

            set2conditions = dict(
                adata.obs.groupby("split").agg({"condition": lambda x: x}).condition
            )
            set2conditions = {i: j.unique().tolist() for i, j in set2conditions.items()}
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys(f"Saved split to {split_path}")

        self.set2conditions = set2conditions

        if split in ("simulation", "simulation_single") and self.subgroup is not None:
            print_sys("Split — test subgroup sizes:")
            for i, j in self.subgroup["test_subgroup"].items():
                print_sys(f"  {i}: {len(j)} perturbation types")
        print_sys("Split ready.")

    def get_dataloader(self, batch_size, test_batch_size=None):
        """Build PyG ``DataLoader``s for train / val / test (GEARS-style grouping)."""
        if test_batch_size is None:
            test_batch_size = batch_size

        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name

        if self.dataset_processed is None:
            raise RuntimeError(
                "dataset_processed is missing. Call pert.load(..., load_cell_graphs=True) before get_dataloader."
            )

        cell_graphs = {}

        if self.split == "no_split":
            i = "test"
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != "ctrl":
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating DataLoaders...")
            test_loader = DataLoader(cell_graphs["test"], batch_size=batch_size, shuffle=False)
            print_sys("DataLoaders ready.")
            return {"test_loader": test_loader}

        else:
            if self.split == "no_test":
                splits = ["train", "val"]
            else:
                splits = ["train", "val", "test"]

            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating DataLoaders...")
            train_loader = DataLoader(cell_graphs["train"], batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(cell_graphs["val"], batch_size=batch_size, shuffle=True)

            if self.split != "no_test":
                test_loader = DataLoader(cell_graphs["test"], batch_size=batch_size, shuffle=False)
                self.dataloader = {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                }
            else:
                self.dataloader = {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                }
            print_sys("DataLoaders ready.")

    def get_pert_idx(self, pert_category):
        """Map perturbation string to indices in ``self.pert_names`` (excluding ctrl)."""
        try:
            pert_idx = [
                np.where(p == self.pert_names)[0][0]
                for p in pert_category.split("+")
                if p != "ctrl"
            ]
        except Exception:
            print(f"Could not parse perturbation: {pert_category}")
            pert_idx = None

        return pert_idx

    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """One cell as a PyG ``Data`` (control profile ``x``, perturbed profile ``y``)."""
        feature_mat = torch.Tensor(X).T
        if pert_idx is None:
            pert_idx = [-1]
        return Data(
            x=feature_mat,
            pert_idx=pert_idx,
            y=torch.Tensor(y),
            de_idx=de_idx,
            pert=pert,
        )

    def create_cell_graph_dataset(self, split_adata, pert_category, num_samples=1):
        """List of ``Data`` graphs for one condition (ctrl pairing for non-ctrl perts)."""
        num_de_genes = 20
        adata_ = split_adata[split_adata.obs["condition"] == pert_category]

        if "rank_genes_groups_cov_all" in adata_.uns:
            de_genes = adata_.uns["rank_genes_groups_cov_all"]
            de = True
        else:
            de = False
            num_de_genes = 1

        Xs = []
        ys = []

        if pert_category != "ctrl":
            pert_idx = self.get_pert_idx(pert_category)

            pert_de_category = adata_.obs["condition_name"][0]
            if de:
                de_idx = np.where(
                    adata_.var_names.isin(np.array(de_genes[pert_de_category][:num_de_genes]))
                )[0]
            else:
                de_idx = [-1] * num_de_genes

            for cell_z in adata_.X:
                ctrl_samples = self.ctrl_adata[
                    np.random.randint(0, len(self.ctrl_adata), num_samples),
                    :,
                ]
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)

        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z)
                ys.append(cell_z)

        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(
                self.create_cell_graph(
                    X.toarray(),
                    y.toarray(),
                    de_idx,
                    pert_category,
                    pert_idx,
                )
            )

        return cell_graphs

    def create_dataset_file(self):
        """Populate ``self.dataset_processed``: condition -> list of cell graphs."""
        print_sys("Building graph dict per condition...")
        self.dataset_processed = {}
        for p in tqdm(self.adata.obs["condition"].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
        print_sys("Graph dict complete.")
