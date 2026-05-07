from __future__ import annotations

import sys
import os
import numpy as np
import pickle
import pandas as pd
from multiprocessing import Pool
import torch
from data_tools.data_utils import tqdm
import networkx as nx
import scipy.sparse as sp


class ProgressBar:
    def __init__(self, total: int, desc: str = "", width: int = 28):
        self.total = max(1, int(total))
        self.desc = desc
        self.width = int(width)
        self.n = 0
        self._last = 0
        self._closed = False

    def _render(self):
        frac = min(1.0, self.n / self.total)
        filled = int(self.width * frac + 0.5)
        bar = "█" * filled + "·" * (self.width - filled)
        msg = f"{self.desc} [{bar}] {self.n:>5d}/{self.total:<5d} ({frac * 100:6.2f}%)"
        sys.stdout.write("\r" + " " * self._last)
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        self._last = len(msg)

    def update(self, inc: int = 1):
        self.n = min(self.total, self.n + int(inc))
        self._render()

    def close(self):
        if not self._closed:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._closed = True


def seed_all(seed: int = 1, deterministic: bool = False) -> None:
    """Set RNG seeds for CPU and CUDA. Optionally enforce deterministic cuDNN (slower)."""
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def report_cuda():
    import torch

    if not torch.cuda.is_available():
        print("[CUDA] No GPU detected; using CPU.")
        return
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"[CUDA] {n} device(s): {names}")


def _parse_gpu_arg(gpu):
    # Accept 'cpu', 'cuda', '0,1,2,3', or a list of ids.
    if isinstance(gpu, (list, tuple)):
        return [int(x) for x in gpu]
    if isinstance(gpu, str):
        g = gpu.strip().lower()
        if g in ("cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"):
            return g
        if "," in g:
            return [int(x) for x in g.split(",") if len(x)]
        try:
            return [int(g)]
        except Exception:
            return g
    return gpu


def setup_gpu_device(gpu_arg: str = "0", gpu_id: int = 0):
    """Select GPU(s) via CUDA_VISIBLE_DEVICES before using torch.

    Ensures single- or multi-GPU behavior is explicit and does not touch unspecified devices.
    """
    import os

    gpu_arg = str(gpu_arg).strip().lower()

    if gpu_arg == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        print("[Device] Using CPU.")
        return torch.device("cpu")

    if "," in gpu_arg:
        ids = [x.strip() for x in gpu_arg.split(",") if x.strip().isdigit()]
    elif gpu_arg.isdigit():
        ids = [gpu_arg]
    elif gpu_arg in ("cuda", "all", ""):
        ids = None
    else:
        raise ValueError(f"Cannot parse gpu argument: {gpu_arg}")

    if ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)

    import torch

    if not torch.cuda.is_available():
        print("[Device] CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")

    n = torch.cuda.device_count()

    if n == 0:
        print("[Device] No usable GPU; using CPU.")
        return torch.device("cpu")

    if n == 1:
        torch.cuda.set_device(0)
        phys = ids[0] if ids else 0
        print(f"[Device] Single GPU: cuda:0 (physical id {phys}).")
        return torch.device("cuda:0")

    print(f"[Device] Multi-GPU: cuda:0..cuda:{n - 1} (physical ids {ids}).")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def use_data_parallel_if_available(model, device_ids=None):
    """Prefer torch_geometric.nn.DataParallel for PyG batches; fall back to torch.nn.DataParallel."""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n <= 1:
        return model
    device_ids = list(range(n)) if device_ids is None else device_ids
    try:
        from torch_geometric.nn import DataParallel as GeoDP

        dp = GeoDP(model, device_ids=device_ids)
        print(f"[DP] torch_geometric.nn.DataParallel, device_count={len(dp.device_ids)}")
        return dp
    except Exception as e:
        print(f"[DP] GeoDP unavailable; using torch.nn.DataParallel: {e}")
        import torch.nn as nn

        dp = nn.DataParallel(model, device_ids=device_ids)
        print(f"[DP] torch.nn.DataParallel, device_count={len(dp.device_ids)}")
        return dp


def pick_device_for_step(step: int, gpu_arg):
    """Optional round-robin device per step (no true parallelism). Mostly for debugging."""
    import torch

    parsed = _parse_gpu_arg(gpu_arg)
    if isinstance(parsed, list) and len(parsed) > 0:
        gid = parsed[step % len(parsed)]
        return torch.device(f"cuda:{gid}")
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def list_available_gpus() -> None:
    """Print all visible CUDA devices."""
    if not torch.cuda.is_available():
        print("No CUDA devices available.")
        return
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s):")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")


def get_similarity_network(
    adata,
    threshold: float,
    k: int,
    data_path: str,
    data_name: str,
    split: str,
    seed: int,
    train_gene_set_size: float,
    set2conditions,
    network_type: str = "go",
    default_pert_graph: bool = True,
    pert_list=None,
    **kwargs,
):
    """Return a DataFrame of edges with columns source, target, weight.

    For network_type='go', use an existing GO edge file if present; otherwise call make_GO.
    Co-expression and other branches follow the original GEARS-style behavior.
    """
    import os
    import tarfile
    import pickle
    import numpy as np
    import pandas as pd

    def _norm_gene(g):
        if g is None:
            return None
        g = str(g)
        g = g.split(".")[0]
        return g.upper()

    if hasattr(adata.var, "gene_name"):
        data_genes = [_norm_gene(x) for x in adata.var.gene_name.tolist()]
    elif "gene_name" in adata.var.columns:
        data_genes = [_norm_gene(x) for x in adata.var["gene_name"].tolist()]
    else:
        data_genes = [_norm_gene(x) for x in adata.var.index.tolist()]
    data_gene_set = set(x for x in data_genes if x)

    if network_type.lower() in {"go", "go-sim", "go_sim", "go_similarity"}:
        base_dir = data_path if data_path is not None else "./data"

        csv_dir = os.path.join(base_dir, "go_essential_all")
        csv_path = os.path.join(csv_dir, "go_essential_all.csv")
        tar_path = os.path.join(base_dir, "go_essential_all.tar.gz")

        os.makedirs(csv_dir, exist_ok=True)
        if (not os.path.isfile(csv_path)) and os.path.isfile(tar_path):
            with tarfile.open(tar_path, "r:gz") as tf:
                member = None
                for m in tf.getmembers():
                    if m.name.endswith("go_essential_all.csv"):
                        member = m
                        break
                if member is None:
                    raise FileNotFoundError(f"go_essential_all.csv not found inside {tar_path}")
                with tf.extractfile(member) as fh, open(csv_path, "wb") as fo:
                    fo.write(fh.read())

        df_csv = None
        if os.path.isfile(csv_path):
            tmp = pd.read_csv(csv_path)
            cols = {c.lower(): c for c in tmp.columns}
            g1 = cols.get("gene1", cols.get("source", None))
            g2 = cols.get("gene2", cols.get("target", None))
            w = cols.get("weight", None)
            if g1 is not None and g2 is not None:
                src = tmp[g1].astype(str).map(_norm_gene)
                tgt = tmp[g2].astype(str).map(_norm_gene)
                weight = (
                    tmp[w].astype(float).values
                    if w is not None
                    else np.ones(len(tmp), dtype=np.float32)
                )
                df_csv = pd.DataFrame({"source": src, "target": tgt, "weight": weight})

        if df_csv is not None and len(df_csv) > 0:
            df_all = df_csv.copy()
            df_all = df_all.loc[
                df_all["source"].isin(data_gene_set) & df_all["target"].isin(data_gene_set)
            ].copy()
            df_all = df_all.loc[df_all["source"] != df_all["target"]]
            uu = np.minimum(df_all["source"].values, df_all["target"].values)
            vv = np.maximum(df_all["source"].values, df_all["target"].values)
            ww = df_all["weight"].astype(float).values
            df_all = pd.DataFrame({"source": uu, "target": vv, "weight": ww}).groupby(
                ["source", "target"], as_index=False
            )["weight"].max()
            print("[GO] Using existing global GO graph (go_essential_all.csv).")
            return df_all.reset_index(drop=True)

        ds_paths = [
            os.path.join(base_dir, f"go_essential_{data_name}.csv"),
            os.path.join(base_dir, data_name, f"go_essential_{data_name}.csv"),
        ]
        for p in ds_paths:
            if os.path.isfile(p):
                tmp = pd.read_csv(p)
                cols = {c.lower(): c for c in tmp.columns}
                g1 = cols.get("source", cols.get("gene1", None))
                g2 = cols.get("target", cols.get("gene2", None))
                w = cols.get("weight", cols.get("importance", None))
                if g1 is not None and g2 is not None:
                    src = tmp[g1].astype(str).map(_norm_gene)
                    tgt = tmp[g2].astype(str).map(_norm_gene)
                    weight = (
                        tmp[w].astype(float).values
                        if w is not None
                        else np.ones(len(tmp), dtype=np.float32)
                    )
                    df_all = pd.DataFrame({"source": src, "target": tgt, "weight": weight})
                    df_all = df_all.loc[
                        df_all["source"].isin(data_gene_set) & df_all["target"].isin(data_gene_set)
                    ].copy()
                    df_all = df_all.loc[df_all["source"] != df_all["target"]]
                    uu = np.minimum(df_all["source"].values, df_all["target"].values)
                    vv = np.maximum(df_all["source"].values, df_all["target"].values)
                    ww = df_all["weight"].astype(float).values
                    df_all = pd.DataFrame({"source": uu, "target": vv, "weight": ww}).groupby(
                        ["source", "target"], as_index=False
                    )["weight"].max()
                    print(f"[GO] Using existing dataset GO graph ({os.path.basename(p)}).")
                    return df_all.reset_index(drop=True)

        if default_pert_graph:
            try:
                from tools.utils import make_GO
            except Exception:
                make_GO = None

            if make_GO is None:
                raise RuntimeError(
                    "make_GO is missing and no GO edge file was found; cannot build GO network."
                )

            if pert_list is None:
                conds = adata.obs["condition"].astype(str).tolist()
                perts = []
                for c in conds:
                    perts.extend([t for t in c.split("+") if t and t != "ctrl"])
                pert_list = sorted(list(set(perts)))

            made = make_GO(base_dir, pert_list, data_name, save=False)
            cols = {c.lower(): c for c in made.columns}
            g1 = cols.get("source", cols.get("gene1", None))
            g2 = cols.get("target", cols.get("gene2", None))
            w = cols.get("weight", cols.get("importance", None))
            if g1 is None or g2 is None:
                raise ValueError("make_GO output must include source/target (or gene1/gene2).")
            src = made[g1].astype(str).map(_norm_gene)
            tgt = made[g2].astype(str).map(_norm_gene)
            weight = (
                made[w].astype(float).values
                if w is not None
                else np.ones(len(made), dtype=np.float32)
            )
            df_all = pd.DataFrame({"source": src, "target": tgt, "weight": weight})
            df_all = df_all.loc[
                df_all["source"].isin(data_gene_set) & df_all["target"].isin(data_gene_set)
            ].copy()
            df_all = df_all.loc[df_all["source"] != df_all["target"]]
            uu = np.minimum(df_all["source"].values, df_all["target"].values)
            vv = np.maximum(df_all["source"].values, df_all["target"].values)
            ww = df_all["weight"].astype(float).values
            df_all = pd.DataFrame({"source": uu, "target": vv, "weight": ww}).groupby(
                ["source", "target"], as_index=False
            )["weight"].max()
            print("[GO] Built custom GO graph (no prior edge file on disk).")
            return df_all.reset_index(drop=True)

        raise RuntimeError(
            "No GO edges available. Provide go_essential_all.csv or go_essential_{data_name}.csv, "
            "or set default_pert_graph=True to build from the perturbation set."
        )

    elif network_type.lower() in {"co-express", "coexpr", "co_expression", "coexpression"}:
        df_out = get_coexpression_network_from_train(
            adata=adata,
            threshold=threshold,
            k=k,
            data_path=data_path,
            data_name=data_name,
            split=split,
            seed=seed,
            train_gene_set_size=train_gene_set_size,
            set2conditions=set2conditions,
        )
        df_out["source"] = df_out["source"].astype(str).map(_norm_gene)
        df_out["target"] = df_out["target"].astype(str).map(_norm_gene)
        df_out = df_out.loc[
            df_out["source"].isin(data_gene_set) & df_out["target"].isin(data_gene_set)
        ].copy()
        df_out = df_out.loc[df_out["source"] != df_out["target"]]
        return df_out.reset_index(drop=True)

    else:
        raise ValueError(f"Unknown network_type: {network_type}")


def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """Build a GO functional similarity network over perturbation genes (multiprocessing).

    Returns:
        pd.DataFrame: edges with endpoints and an importance/weight column.
    """
    fname = "./data/go_essential_" + data_name + ".csv"
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, "gene2go_all.pkl"), "rb") as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list}

    print("Building custom GO graph (may take several minutes)...")
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(
                p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                total=len(gene2go.keys()),
            )
        )
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(columns={0: "source", 1: "target", 2: "importance"})
    if save:
        print("Saving edge list to file.")
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list


def get_coexpression_network_from_train(
    adata,
    threshold=0.0,
    k=None,
    data_path=None,
    data_name=None,
    split=None,
    seed=0,
    train_gene_set_size=None,
    set2conditions=None,
):
    """Coexpression network from training cells; endpoints are gene symbols aligned to node_map.

    If there are too few samples, returns self-loop placeholders per gene.
    Otherwise uses Pearson correlation with threshold or top-k; importance is |corr|.
    """

    if hasattr(adata, "obs") and "split" in adata.obs:
        train_mask = adata.obs["split"].astype(str).str.lower() == "train"
        if train_mask.sum() == 0:
            train_mask[:] = True
    else:
        train_mask = np.ones(adata.n_obs, dtype=bool)

    X = adata.X[train_mask]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    def _norm_gene(g):
        g = str(g).split(".")[0]
        return g.upper()

    if hasattr(adata.var, "gene_name"):
        gene_names = [_norm_gene(x) for x in adata.var.gene_name.tolist()]
    elif "gene_name" in adata.var.columns:
        gene_names = [_norm_gene(x) for x in adata.var["gene_name"].tolist()]
    else:
        gene_names = [_norm_gene(x) for x in adata.var.index.tolist()]
    gene_names = np.asarray(gene_names, dtype=object)

    N, G = (int(X.shape[0]), int(X.shape[1])) if X.size > 0 else (0, int(getattr(adata, "n_vars", 0)))

    def _return_self_loops_by_name() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source": gene_names[:G],
                "target": gene_names[:G],
                "importance": np.ones(G, dtype=np.float32),
            }
        )

    if N < 2 or G == 0:
        return _return_self_loops_by_name()

    C = np_pearson_cor(X, X)
    if C.size == 0:
        return _return_self_loops_by_name()
    np.fill_diagonal(C, 0.0)
    absC = np.abs(C)

    edges = []
    if k is None or k <= 0:
        threshold = 0.0 if threshold is None else float(threshold)
        src, dst = np.where(absC >= threshold)
        if src.size == 0:
            return _return_self_loops_by_name()
        w = absC[src, dst].astype(np.float32)
        edges = np.stack([src, dst, w], axis=1)
    else:
        k = int(k)
        for i in range(G):
            row = absC[i]
            idx = np.argpartition(-row, kth=min(k, G - 1) - 1)[:k]
            idx = idx[idx != i]
            if idx.size == 0:
                continue
            w = row[idx].astype(np.float32)
            src = np.full_like(idx, i)
            edges.append(np.stack([src, idx, w], axis=1))
        if len(edges) == 0:
            return _return_self_loops_by_name()
        edges = np.concatenate(edges, axis=0)

    src_idx = edges[:, 0].astype(int)
    dst_idx = edges[:, 1].astype(int)
    w = edges[:, 2].astype(np.float32)

    src_name = gene_names[src_idx]
    dst_name = gene_names[dst_idx]

    df = pd.DataFrame({"source": src_name, "target": dst_name, "importance": w})
    df["importance"] = np.nan_to_num(df["importance"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(df) == 0:
        return _return_self_loops_by_name()
    return df


def get_GO_edge_list(args):
    """Worker for make_GO: Jaccard similarity over GO term sets for one gene vs all others."""
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list


def np_pearson_cor(x, y):
    """Column-wise Pearson correlation; accepts numpy, scipy.sparse, torch, or pandas.

    For fewer than two samples, returns an identity matrix (caller may zero the diagonal).
    """

    def _to_2d_dense(a):
        if a is None:
            return None
        if sp.issparse(a):
            a = a.toarray()
        try:
            import torch

            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
        except Exception:
            pass
        try:
            import pandas as pd

            if isinstance(a, (pd.DataFrame, pd.Series)):
                a = a.to_numpy()
        except Exception:
            pass
        a = np.asarray(a)
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a.reshape(1, -1)
        return a

    x = _to_2d_dense(x)
    y = _to_2d_dense(y)

    if x is None or y is None or x.size == 0 or y.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if x.shape[0] < 2 or y.shape[0] < 2:
        G = x.shape[1]
        return np.eye(G, dtype=np.float32)

    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    xv = x - x.mean(axis=0, keepdims=True)
    yv = y - y.mean(axis=0, keepdims=True)

    denom = (np.linalg.norm(xv, axis=0) * np.linalg.norm(yv, axis=0)) + 1e-12
    corr = (xv.T @ yv) / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


class GeneSimNetwork:
    def __init__(
        self,
        edge_df,
        pert_list=None,
        node_map=None,
        weight_col=None,
        keep_self_loops: bool = False,
    ):
        import numpy as np
        import pandas as pd
        import torch
        import networkx as nx

        df = edge_df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]

        cand_src = ["source", "src", "gene1", "from", "u"]
        cand_tgt = ["target", "dst", "gene2", "to", "v"]
        src_col = next((c for c in cand_src if c in df.columns), None)
        tgt_col = next((c for c in cand_tgt if c in df.columns), None)
        if src_col is None or tgt_col is None:
            raise ValueError(
                f"[GeneSimNetwork] Cannot infer endpoint columns. Found: {list(df.columns)}"
            )

        cand_w = [weight_col, "importance", "weight", "score", "sim", "similarity", "edge_weight", "w"]
        cand_w = [c for c in cand_w if c]
        w_col = next((c for c in cand_w if c in df.columns), None)
        if w_col is None:
            df["weight"] = 1.0
        else:
            df["weight"] = pd.to_numeric(df[w_col], errors="coerce").fillna(0.0)

        df = df[[src_col, tgt_col, "weight"]].rename(columns={src_col: "source", tgt_col: "target"})

        def _norm(x) -> str:
            s = str(x).strip()
            if not s:
                return s
            s = s.split(".")[0]
            return s.upper()

        if node_map is not None:
            node_map = {_norm(k): int(v) for k, v in node_map.items()}

        src_key = df["source"].map(_norm)
        tgt_key = df["target"].map(_norm)

        if node_map is not None:
            src_id = src_key.map(node_map)
            tgt_id = tgt_key.map(node_map)
            valid = src_id.notna() & tgt_id.notna()

            if not bool(valid.any()):
                s_int = pd.to_numeric(df["source"], errors="coerce")
                t_int = pd.to_numeric(df["target"], errors="coerce")
                if s_int.notna().all() and t_int.notna().all():
                    s_int = s_int.astype(np.int64)
                    t_int = t_int.astype(np.int64)
                    n_nodes = int(max(node_map.values())) + 1
                    valid = (s_int >= 0) & (s_int < n_nodes) & (t_int >= 0) & (t_int < n_nodes)
                    src_id = s_int[valid]
                    tgt_id = t_int[valid]
        else:
            genes = pd.Index(pd.unique(pd.concat([src_key, tgt_key], ignore_index=True)))
            node_map = {g: i for i, g in enumerate(genes)}
            src_id = src_key.map(node_map)
            tgt_id = tgt_key.map(node_map)
            valid = src_id.notna() & tgt_id.notna()

        if not bool(valid.any()):
            raise ValueError(
                "[GeneSimNetwork] No valid edges after filtering; check gene names against node_map."
            )

        w = df.loc[valid, "weight"].astype(np.float32).to_numpy()
        if isinstance(src_id, pd.Series) and len(src_id) == len(df):
            src_arr = src_id.loc[valid].astype(np.int64).to_numpy()
            tgt_arr = tgt_id.loc[valid].astype(np.int64).to_numpy()
        else:
            src_arr = np.asarray(src_id, dtype=np.int64)
            tgt_arr = np.asarray(tgt_id, dtype=np.int64)

        if not keep_self_loops:
            m = src_arr != tgt_arr
            src_arr, tgt_arr, w = src_arr[m], tgt_arr[m], w[m]

        self.edge_index = torch.from_numpy(np.vstack([src_arr, tgt_arr])).long()
        self.edge_weight = torch.from_numpy(w).float()

        self.G = nx.Graph()
        self.G.add_weighted_edges_from([(int(u), int(v), float(ww)) for u, v, ww in zip(src_arr, tgt_arr, w)])

        few = min(5, self.edge_index.shape[1])
        print(
            "[GeneSimNetwork] weight column:",
            (w_col if w_col is not None else "weight"),
            "| #edges=",
            self.edge_index.shape[1],
            "| #nodes=",
            len(node_map),
        )
        if few > 0:
            ex = list(
                zip(
                    self.edge_index[0, :few].tolist(),
                    self.edge_index[1, :few].tolist(),
                    [round(float(x), 4) for x in self.edge_weight[:few].tolist()],
                )
            )
            print(f"[GeneSimNetwork] sample edges (u, v, w), first {few}: {ex}")
