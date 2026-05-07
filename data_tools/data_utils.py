"""Utilities for GEARS-style perturbation data: downloads, DE genes, and splitting.

Train/validation/test splits and subgroup logic match the ``gears.PertData`` / ``DataSplitter``
conventions used in the original benchmark implementation:

  https://github.com/snap-stanford/GEARS

See ``PertData.py`` module docstring for a concise list of split modes
(``simulation``, ``combo_seen*``, etc.).
"""
import scanpy as sc

sc.settings.verbosity = 0
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import requests
from zipfile import ZipFile
import tarfile

warnings.filterwarnings("ignore")


def print_sys(s):
    """Print to stderr with flush (useful in multiprocessing)."""
    print(s, flush=True, file=sys.stderr)


def dataverse_download(url, save_path):
    """Download a file from Dataverse (streaming)."""
    if os.path.exists(save_path):
        print_sys("Local file already exists; skipping download.")
    else:
        print_sys("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def zip_data_download_wrapper(url, save_path, data_path):
    """Download a zip archive and extract it under ``data_path``."""
    if os.path.exists(save_path):
        print_sys("Local folder already exists; skipping download.")
    else:
        dataverse_download(url, save_path + ".zip")
        print_sys("Extracting zip...")
        with ZipFile((save_path + ".zip"), "r") as zip:
            zip.extractall(path=data_path)
        print_sys("Extraction done.")


def tar_data_download_wrapper(url, save_path, data_path):
    """Download a tar.gz archive and extract it under ``data_path``."""
    if os.path.exists(save_path):
        print_sys("Local folder already exists; skipping download.")
    else:
        dataverse_download(url, save_path + ".tar.gz")
        print_sys("Extracting tar.gz...")
        with tarfile.open(save_path + ".tar.gz") as tar:
            tar.extractall(path=data_path)
        print_sys("Extraction done.")


def parse_single_pert(i):
    """Parse a single-gene perturbation label (e.g. ``ctrl+GENE`` -> ``GENE``)."""
    a = i.split("+")[0]
    b = i.split("+")[1]
    if a == "ctrl":
        pert = b
    else:
        pert = a
    return pert


def parse_combo_pert(i):
    """Parse a double-gene perturbation (e.g. ``A+B``)."""
    return i.split("+")[0], i.split("+")[1]


def combine_res(res_1, res_2):
    """Concatenate numpy arrays per key."""
    res_out = {}
    for key in res_1:
        res_out[key] = np.concatenate([res_1[key], res_2[key]])
    return res_out


def parse_any_pert(p):
    """Return constituent gene names for single- or double-perturbation strings."""
    if ("ctrl" in p) and (p != "ctrl"):
        return [parse_single_pert(p)]
    elif "ctrl" not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]


def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):
    """Per covariate level (e.g. cell type), rank genes vs. control with scanpy.

    Stores a dict of DE gene names per group in ``adata.uns[key_added]``.
    """
    gene_dict = {}

    cov_categories = adata.obs[covariate].unique()

    for cov_cat in cov_categories:
        control_group_cov = "_".join([cov_cat, control_group])

        adata_cov = adata[adata.obs[covariate] == cov_cat]

        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False,
        )

        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict


def get_DE_genes(adata, skip_calc_de):
    """Add GEARS-style obs columns and run covariate-aware DE (unless skipped)."""
    adata.obs.loc[:, "dose_val"] = adata.obs.condition.apply(
        lambda x: "1+1" if len(x.split("+")) == 2 else "1"
    )

    adata.obs.loc[:, "control"] = adata.obs.condition.apply(
        lambda x: 0 if len(x.split("+")) == 2 else 1
    )

    adata.obs.loc[:, "condition_name"] = adata.obs.apply(
        lambda x: "_".join([x.cell_type, x.condition, x.dose_val]), axis=1
    )

    adata.obs = adata.obs.astype("category")

    if not skip_calc_de:
        rank_genes_groups_by_cov(
            adata,
            groupby="condition_name",
            covariate="cell_type",
            control_group="ctrl_1",
            n_genes=len(adata.var),
            key_added="rank_genes_groups_cov_all",
        )
    return adata


def get_dropout_non_zero_genes(adata):
    """Tag top DE genes with non-dropout / non-zero heuristics; fill ``adata.uns`` for GEARS.

    Zeros in scRNA-seq may be dropout; genes that are zero in both pert and ctrl
    (``true_zeros``) are treated as reliable negatives when shortlisting DE sets.
    """
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for cond in unique_conditions:
        conditions2index[cond] = np.where(adata.obs.condition == cond)[0]

    condition2mean_expression = {}
    for cond, idx in conditions2index.items():
        condition2mean_expression[cond] = np.mean(adata.X[idx], axis=0)

    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(
        len(unique_conditions), adata.X.toarray().shape[1]
    )
    ctrl_mean = mean_expression[np.where(pert_list == "ctrl")[0]]

    pert2pert_full_id = dict(adata.obs[["condition", "condition_name"]].values)
    pert_full_id2pert = dict(adata.obs[["condition_name", "condition"]].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns["rank_genes_groups_cov_all"].keys():
        p = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == p].X, axis=0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]

        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl_mean)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top_de_genes = adata.uns["rank_genes_groups_cov_all"][pert]
        gene_idx_top = [gene_id2idx[gene] for gene in top_de_genes]

        non_dropout_20 = [idx for idx in gene_idx_top if idx in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[idx] for idx in non_dropout_20]

        non_zero_20 = [idx for idx in gene_idx_top if idx in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[idx] for idx in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)

    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl_mean)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))

    adata.uns["top_non_dropout_de_20"] = top_non_dropout_de_20
    adata.uns["non_dropout_gene_idx"] = non_dropout_gene_idx
    adata.uns["non_zeros_gene_idx"] = non_zeros_gene_idx
    adata.uns["top_non_zero_de_20"] = top_non_zero_de_20

    return adata


class DataSplitter:
    """GEARS-compatible train/val/test splits over perturbation identifiers.

    Implements ``simulation`` (gene holdout + combo buckets), ``simulation_single``,
    ``combo_seen*``, ``single``, and ``no_test`` routing used by ``PertData.prepare_split``.
    """

    def __init__(self, adata, split_type="single", seen=0):
        """Args:
        adata: AnnData with ``obs['condition']`` perturbation labels.
        split_type: One of simulation / simulation_single / combo / single / no_test / ...
        seen: For ``combo`` splits, number (0/1/2) of genes seen at train time.
        """
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(
        self,
        test_size=0.1,
        test_pert_genes=None,
        test_perts=None,
        split_name="split",
        seed=None,
        val_size=0.1,
        train_gene_set_size=0.75,
        combo_seen2_train_frac=0.75,
        only_test_set_perts=False,
    ):
        """Write ``adata.obs[split_name]`` in {'train','val','test'}; may return subgroup dict."""
        np.random.seed(seed=seed)

        unique_perts = [p for p in self.adata.obs["condition"].unique() if p != "ctrl"]

        if self.split_type == "simulation":
            train, test, test_subgroup = self.get_simulation_split(
                unique_perts,
                train_gene_set_size,
                combo_seen2_train_frac,
                seed,
                test_perts,
                only_test_set_perts,
            )
            train, val, val_subgroup = self.get_simulation_split(train, 0.9, 0.9, seed)
            train.append("ctrl")

        elif self.split_type == "simulation_single":
            train, test, test_subgroup = self.get_simulation_split_single(
                unique_perts, train_gene_set_size, seed, test_perts, only_test_set_perts
            )
            train, val, val_subgroup = self.get_simulation_split_single(train, 0.9, seed)

        elif self.split_type == "no_test":
            train, val = self.get_split_list(unique_perts, test_size=val_size)

        else:
            train, test = self.get_split_list(
                unique_perts,
                test_pert_genes=test_pert_genes,
                test_perts=test_perts,
                test_size=test_size,
            )
            train, val = self.get_split_list(train, test_size=val_size)

        map_dict = {x: "train" for x in train}
        map_dict.update({x: "val" for x in val})
        if self.split_type != "no_test":
            map_dict.update({x: "test" for x in test})
        map_dict.update({"ctrl": "train"})

        self.adata.obs[split_name] = self.adata.obs["condition"].map(map_dict)

        if self.split_type == "simulation":
            return self.adata, {
                "test_subgroup": test_subgroup,
                "val_subgroup": val_subgroup,
            }
        if self.split_type == "simulation_single":
            return self.adata, {
                "test_subgroup": test_subgroup,
                "val_subgroup": val_subgroup,
            }
        return self.adata

    def get_simulation_split_single(
        self, pert_list, train_gene_set_size=0.85, seed=1, test_set_perts=None, only_test_set_perts=False
    ):
        """Gene holdout for single-gene perts only; test = perts whose gene was not in train pool."""
        unique_pert_genes = self.get_genes_from_perts(pert_list)

        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)

        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            train_gene_candidates = np.random.choice(
                unique_pert_genes,
                int(len(unique_pert_genes) * train_gene_set_size),
                replace=False,
            )

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[
                    ~np.isin(train_gene_candidates, test_set_perts)
                ]
                ood_genes_exclude_test_set = np.setdiff1d(
                    unique_pert_genes,
                    np.union1d(train_gene_candidates, test_set_perts),
                )
                train_set_addition = np.random.choice(
                    ood_genes_exclude_test_set, num_overlap, replace=False
                )
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))

            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)

        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list, "single")
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, "single")

        assert len(unseen_single) + len(pert_single_train) == len(pert_list)

        return pert_single_train, unseen_single, {"unseen_single": unseen_single}

    def get_simulation_split(
        self,
        pert_list,
        train_gene_set_size=0.85,
        combo_seen2_train_frac=0.85,
        seed=1,
        test_set_perts=None,
        only_test_set_perts=False,
    ):
        """Full ``simulation`` split: gene holdout plus combo_seen0/1/2 and unseen_single buckets."""
        unique_pert_genes = self.get_genes_from_perts(pert_list)

        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)

        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            train_gene_candidates = np.random.choice(
                unique_pert_genes,
                int(len(unique_pert_genes) * train_gene_set_size),
                replace=False,
            )

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[
                    ~np.isin(train_gene_candidates, test_set_perts)
                ]
                ood_genes_exclude_test_set = np.setdiff1d(
                    unique_pert_genes,
                    np.union1d(train_gene_candidates, test_set_perts),
                )
                train_set_addition = np.random.choice(
                    ood_genes_exclude_test_set, num_overlap, replace=False
                )
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))

            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)

        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list, "single")
        pert_train.extend(pert_single_train)

        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list, "combo")

        combo_seen1 = [
            x
            for x in pert_combo
            if len([t for t in x.split("+") if t in train_gene_candidates]) == 1
        ]
        pert_test.extend(combo_seen1)

        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(
            pert_combo,
            int(len(pert_combo) * combo_seen2_train_frac),
            replace=False,
        )

        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)

        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, "single")
        pert_test.extend(unseen_single)

        combo_seen0 = [
            x
            for x in self.get_perts_from_genes(ood_genes, pert_list, "combo")
            if len([t for t in x.split("+") if t in train_gene_candidates]) == 0
        ]
        pert_test.extend(combo_seen0)

        assert (
            len(combo_seen1)
            + len(combo_seen0)
            + len(unseen_single)
            + len(pert_train)
            + len(combo_seen2)
            == len(pert_list)
        )

        return pert_train, pert_test, {
            "combo_seen0": combo_seen0,
            "combo_seen1": combo_seen1,
            "combo_seen2": combo_seen2,
            "unseen_single": unseen_single,
        }

    def get_split_list(self, pert_list, test_size=0.1, test_pert_genes=None, test_perts=None, hold_outs=True):
        """Split ``pert_list`` into train and test lists (used by ``single`` / ``combo`` / ``no_test``)."""
        single_perts = [p for p in pert_list if "ctrl" in p and p != "ctrl"]
        combo_perts = [p for p in pert_list if "ctrl" not in p]
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        hold_out = []

        if test_pert_genes is None:
            test_pert_genes = np.random.choice(
                unique_pert_genes,
                int(len(single_perts) * test_size),
            )

        if self.split_type == "single" or self.split_type == "single_only":
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list, "single")
            if self.split_type == "single_only":
                hold_out = combo_perts
            else:
                hold_out = self.get_perts_from_genes(test_pert_genes, pert_list, "combo")

        elif self.split_type == "no_test":
            if test_perts is None:
                test_perts = np.random.choice(pert_list, int(len(pert_list) * test_size))

        elif self.split_type == "combo":
            if self.seen == 0:
                single_perts = self.get_perts_from_genes(test_pert_genes, pert_list, "single")
                combo_perts = self.get_perts_from_genes(test_pert_genes, pert_list, "combo")
                if hold_outs:
                    hold_out = [
                        t
                        for t in combo_perts
                        if len([x for x in t.split("+") if x not in test_pert_genes]) > 0
                    ]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 1:
                single_perts = self.get_perts_from_genes(test_pert_genes, pert_list, "single")
                combo_perts = self.get_perts_from_genes(test_pert_genes, pert_list, "combo")
                if hold_outs:
                    hold_out = [
                        t
                        for t in combo_perts
                        if len([x for x in t.split("+") if x not in test_pert_genes]) > 1
                    ]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 2:
                if test_perts is None:
                    test_perts = np.random.choice(
                        combo_perts,
                        int(len(combo_perts) * test_size),
                    )
                else:
                    test_perts = np.array(test_perts)
        else:
            if test_perts is None:
                test_perts = np.random.choice(
                    combo_perts,
                    int(len(combo_perts) * test_size),
                )

        train_perts = [p for p in pert_list if (p not in test_perts) and (p not in hold_out)]
        return train_perts, test_perts

    def get_perts_from_genes(self, genes, pert_list, type_="both"):
        """All perturbations in ``pert_list`` that involve any of ``genes`` (single/combo/both)."""
        single_perts = [p for p in pert_list if ("ctrl" in p) and (p != "ctrl")]
        combo_perts = [p for p in pert_list if "ctrl" not in p]

        perts = []

        if type_ == "single":
            pert_candidate_list = single_perts
        elif type_ == "combo":
            pert_candidate_list = combo_perts
        elif type_ == "both":
            pert_candidate_list = pert_list

        for p in pert_candidate_list:
            for g in genes:
                if g in parse_any_pert(p):
                    perts.append(p)
                    break
        return perts

    def get_genes_from_perts(self, perts):
        """Unique gene symbols (excluding ctrl) appearing in perturbation strings."""
        if type(perts) is str:
            perts = [perts]
        gene_list = [p.split("+") for p in np.unique(perts)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != "ctrl"]
        return np.unique(gene_list)


def get_genes_from_perts(perts):
    """Module helper: unique genes (excluding ctrl) from perturbation labels."""
    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split("+") for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != "ctrl"]
    return list(np.unique(gene_list))


def filter_pert_in_go(condition, pert_names):
    """Return True if both alleles are ctrl or appear in ``pert_names`` (GO-supported graph)."""
    if condition == "ctrl":
        return True
    else:
        cond1 = condition.split("+")[0]
        cond2 = condition.split("+")[1]
        num_ctrl = (cond1 == "ctrl") + (cond2 == "ctrl")
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        if num_ctrl + num_in_perts == 2:
            return True
        else:
            return False

