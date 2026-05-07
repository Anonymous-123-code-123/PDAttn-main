"""Perturbation–gene graph construction for PDAttn (GO-based message passing).

The **perturbation-side GO graph** (edges among perturbed targets from Gene Ontology
similarity, then consumed by graph convolution in ``PDAttnModel``) follows the same
design pattern as **GEARS**: build a functional similarity graph over perturbation genes
and feed it into a GNN backbone. We adapt that workflow here via
``get_similarity_network(..., network_type='go')`` and ``GeneSimNetwork``. Reference:

  https://github.com/snap-stanford/GEARS

This module does **not** reimplement GEARS’ full model; it only constructs ``edge_index``
/ ``edge_weight`` for the perturbation graph and packages ``make_model_args`` for
``PDAttnModel``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from tools.utils import GeneSimNetwork, get_similarity_network


class GenePertGraphBuilder:
    """Build and cache the GO similarity graph over perturbation genes (PyG format)."""

    def __init__(
        self,
        adata,
        gene_list: List[str],
        pert_list: List[str],
        node_map: Dict[str, int],
        node_map_pert: Dict[str, int],
        data_path: str,
        dataset_name: str,
        split: str,
        seed: int,
        train_gene_set_size: float,
        set2conditions: Dict[str, List[str]],
        default_pert_graph: bool = True,
        k_go: int = 20,
        coexpr_threshold: float = 0.4,
        device: Optional[torch.device] = None,
    ):
        self.adata = adata
        self.gene_list = gene_list
        self.pert_list = pert_list
        self.node_map = node_map
        self.node_map_pert = node_map_pert
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        self.seed = seed
        self.train_gene_set_size = train_gene_set_size
        self.set2conditions = set2conditions
        self.default_pert_graph = default_pert_graph
        self.k_go = int(k_go)
        self.coexpr_threshold = float(coexpr_threshold)
        self.device = device if device is not None else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.G_go = None
        self.G_go_weight = None

    def _build_go_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(edge_index [2, E], edge_weight [E])`` for the GO perturbation graph.

        Loads or builds edge table via ``get_similarity_network(network_type="go")``,
        then maps endpoints to integer node ids with ``GeneSimNetwork``.
        """
        df_go = get_similarity_network(
            adata=self.adata,
            threshold=self.coexpr_threshold,
            k=self.k_go,
            data_path=self.data_path,
            data_name=self.dataset_name,
            split=self.split,
            seed=self.seed,
            train_gene_set_size=self.train_gene_set_size,
            set2conditions=self.set2conditions,
            network_type="go",
            default_pert_graph=self.default_pert_graph,
            pert_list=self.pert_list,
        )
        go_net = GeneSimNetwork(df_go, self.pert_list, node_map=self.node_map_pert)
        edge_index = go_net.edge_index
        edge_weight = go_net.edge_weight
        return edge_index, edge_weight

    def build_all(self) -> Dict[str, torch.Tensor]:
        """Materialize GO graph tensors and return a dict with counts for the model.

        Keys: ``G_go``, ``G_go_weight``, ``num_genes``, ``num_perts``.
        """
        G_go, G_go_w = self._build_go_graph()

        self.G_go = G_go.to(self.device)
        self.G_go_weight = G_go_w.to(self.device)

        return dict(
            G_go=self.G_go,
            G_go_weight=self.G_go_weight,
            num_genes=len(self.gene_list),
            num_perts=len(self.pert_list),
        )

    def make_model_args(
        self,
        hidden_size: int = 64,
        num_go_gnn_layers: int = 1,
        num_attention_heads: int = 4,
        decoder_hidden_size: int = 16,
        no_perturb: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble kwargs for ``PDAttnModel`` (graph, sizes, and pert→gene mask).

        If the graph has not been built yet, ``build_all()`` is called. Dimensions are
        passed through unchanged (no hidden upscaling here).

        Adds ``pert_to_gene`` [num_perts, num_genes]: binary mask linking each perturbation
        name to target gene columns (splits names on ``+`` / ``,`` / ``;``).
        """
        if self.G_go is None:
            _ = self.build_all()

        args = dict(
            G_go=self.G_go,
            G_go_weight=self.G_go_weight,
            num_genes=len(self.gene_list),
            num_perts=len(self.pert_list),
            hidden_size=int(hidden_size),
            num_go_gnn_layers=int(num_go_gnn_layers),
            num_attention_heads=int(num_attention_heads),
            decoder_hidden_size=int(decoder_hidden_size),
            no_perturb=bool(no_perturb),
        )

        p2g = torch.full((len(self.pert_list), len(self.gene_list)), 0.0, dtype=torch.float32)
        g2idx = {g: i for i, g in enumerate(self.gene_list)}
        for p_id, pname in enumerate(self.pert_list):
            genes = [
                t
                for s in str(pname).replace("+", ",").replace(";", ",").split(",")
                for t in [s.strip()]
                if t
            ]
            for g in genes:
                if g in g2idx:
                    p2g[p_id, g2idx[g]] = 1.0
        args["pert_to_gene"] = p2g

        if extra:
            args.update(extra)
        return args
