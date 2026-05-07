# PDAttn: Perturbation-Specific Expression Prediction

## 1. Overview

**PDAttn** predicts gene expression under genetic perturbations in single-cell settings. It uses graph structure related to perturbations (including Gene Ontology, GO) and integrates perturbation signals at **two levels**:

- **Level 1 — gene-level conditioning:** perturbation features **modulate** gene representations (FiLM-style conditioning) so the model encodes **perturbation-aligned, gene-resolution** responses.
- **Level 2 — context-aware injection:** perturbation features are combined with **cell background** information to form a context vector, which is injected into gene representations to capture **perturbation effects that vary across cellular contexts**.

![PDAttn architecture](img/PDAttn%20Model.png)

For splits and PyG dataloaders, this codebase follows conventions aligned with **GEARS**, enabling fair comparison with existing perturbation baselines: [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS).

---

## 2. Requirements

- **Python:** 3.9 or newer recommended.
- **Dependencies:** see [`requirements.txt`](requirements.txt) in the repository root.

**Suggested setup** (run from the repository root, e.g. `PDAttn-main`):

```bash
cd PDAttn-main
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
```

For **GPU** runs, install a CUDA-matched **PyTorch** build first ([PyTorch — Get Started](https://pytorch.org/get-started/locally/)), then install the rest:

```bash
pip install -r requirements.txt
```

**CPU-only** installs can use the same command:

```bash
pip install -r requirements.txt
```

If `torch-geometric` fails to install, follow [PyTorch Geometric — Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and pick the wheel index matching your PyTorch and CUDA version.

---

## 3. Core configuration and hyperparameters

The values below match the defaults in **`task/train_model.py`**, **`PDAttn/PDATT.py`**, and the evaluation script. **Use the same split mode and random seed for training and evaluation.**

### 3.1 Training — `task/train_model.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Root directory for `PertData` |
| `--dataset` | `norman` | Dataset key for `PertData.load`; comma-separated list trains multiple datasets |
| `--epochs` | `15` | Number of training epochs |
| `--lr` | `1e-4` | AdamW learning rate |
| `--batch_size` | `128` | Training batch size |
| `--test_batch_size` | `128` | Validation / test DataLoader batch size |
| `--seed` | `42` | Random seed (splits and training) |
| `--gpu` | `0` | GPU id(s); `cpu` for CPU only; multi-GPU example: `0,1` |
| **Split (fixed in code)** | `split='simulation'` | GEARS-style `simulation` split |
| **`train_gene_set_size`** | `0.75` | Fraction of perturbation genes in the training set (`simulation`) |

**Graph construction** (`GenePertGraphBuilder`, as invoked in `train_model.py`): `k_go=20`, `coexpr_threshold=0.4`.

**Optimizer:** AdamW with `weight_decay=1e-5` (see `train_single_dataset`).

**Checkpoints:** written by default to `model/<dataset_name>.pt` (`model_state`, `C_ctrl`, `mu_ref` / tensors used with `O_pert` in metrics, `args`, validation metrics, etc.).

### 3.2 Model defaults — `build_pdattn_model` (when not overridden by the checkpoint)

| Setting | Default |
|---------|---------|
| `hidden_dim` / `hidden_size` | `64` |
| `num_attention_heads` | `4` |
| `dropout` | `0.1` |
| `decoder_hidden_size` | `16` |

If inference architecture must match training, use the `override_*` flags in `task/analysis_pearson.py` or the `args` stored in the checkpoint.

### 3.3 Inference and evaluation — `task/analysis_pearson.py`

| Argument | Default | Notes |
|----------|---------|-------|
| `--split` | `simulation` | Must match training |
| `--seed` | `42` | Must match training |
| `--batch_size` | `128` | Evaluation batch size |
| `prepare_split` · `train_gene_set_size` | `0.75` | Must match training |

### 3.4 Bulk data preparation — `task/data_load.py` (optional)

Runs a fixed list of datasets through download / preprocessing and split construction. It uses **`prepare_split` with `seed=1`**, **train batch size 32**, and **test batch size 128**. If you combine this script with `train_model.py` (`seed=42`, batch 128), reconcile seeds and batch sizes so splits and logs stay consistent.

### 3.5 Linear baseline — `task/gslm.py` (optional)

| Argument | Default |
|----------|---------|
| `--split` | `simulation` |
| `--seed` | `42` |
| `--train_gene_set_size` | `0.75` |
| `--pca_dim` | `64` |
| `--alpha` (Ridge) | `0.1` |

---

## 4. Datasets

We use public perturbation single-cell datasets; keys match `PertData.load`. NCBI GEO accessions and sources are listed below (links are clickable).

| `data_name` in code | Reference | Link |
|---------------------|-----------|------|
| `norman` | Norman et al., **GSE133344** | [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) |
| `dixit` | Dixit et al., **GSE90063** | [GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063) |
| `adamson` | Adamson et al., **GSE90546** | [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546) |
| `replogle_k562_essential` | Replogle et al., **GSE146194** (K562 essentials subset) | [GSE146194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146194) |
| `replogle_rpe1_essential` | Replogle et al., **GSE146194** (RPE1 essentials subset) | [GSE146194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146194) |
| `xu_kinetics_2024` | Xu et al., **GSE218566** | [GSE218566](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218566) |
| `tiankampmann2021_crispra` / `tiankampmann2021_crispri` | Tian / Kampmann et al. (CRISPRa / CRISPRi) | [Zenodo 13350497](https://zenodo.org/records/13350497) |

GEO series landing pages: [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344), [GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063), [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546), [GSE146194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146194), [GSE218566](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218566).

---

## 5. Repository layout and usage

Run the commands below from the repository root so that the `PDAttn`, `data_tools`, `tools`, and `task` packages resolve (or add the root to `PYTHONPATH`).

| Path | Role |
|------|------|
| `data_tools/PertData.py` | Data loading, GEARS-style splits, PyG `DataLoader` |
| `task/data_load.py` | (Optional) bulk download / preprocessing, splits, dataloaders |
| `task/train_model.py` | **Train** PDAttn; saves `model/<dataset>.pt` |
| `task/analysis_pearson.py` | **Inference and test-set evaluation** (Pearson, subgroups, top-DE, Systema-related metrics, etc.) |
| `task/gslm.py` | (Optional) gene-specific linear baseline — train and evaluate |

**Examples:**

```bash
# Train (single dataset)
python -m task.train_model --dataset norman --data_dir ./data --gpu 0 --epochs 15 --seed 42

# Test-set evaluation (split and seed must match training).
# --ckpt is optional: if omitted, loads model/<dataset>.pt (e.g. model/norman.pt).
python -m task.analysis_pearson --dataset norman --data_dir ./data --gpu 0 --seed 42 --split simulation
```

From the repository root, `python -m task.<module>` requires this directory on `PYTHONPATH` (the current working directory is enough when you `cd` here). Nested packages (`PDAttn`, `data_tools`, `tools`) do not ship `__init__.py` files; this is valid on **Python 3.3+** via [namespace packages](https://docs.python.org/3/reference/import.html#namespace-packages).

---

## 6. Processed data, weights, and split details

Full preprocessing documentation, **split specifications** matching the paper, and **model checkpoints** will be distributed via **links provided in the paper** once it is officially published. This repository focuses on reproducible scripts and default hyperparameters so experiments can be rebuilt from public sources locally.

---

## Citation

A BibTeX entry and DOI will be added here after the manuscript is published.

---

## 7. Acknowledgements

We thank the authors and curators of the public datasets cited above (Norman, Dixit, Adamson, Replogle, Xu, Tian / Kampmann, and colleagues), and platforms such as GEO and Zenodo for open access. We also acknowledge community efforts such as **GEARS** for standardized splits and data practices that facilitate comparison and reproduction.
