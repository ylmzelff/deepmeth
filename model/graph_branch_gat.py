"""
GATv2-based Hi-C graph branch - replaces model/graph_branch.py's
GCN_Structure (plain 3-layer Kipf & Welling GCN) after standalone
diagnosis showed it barely learning anything (val_mcc ~0.26 alone, versus
the sequence branch's ~0.65 and the full fused model's ~0.66 - a graph
branch this weak was contributing almost nothing to the fusion). Root
issues identified in model/graph_branch.py:
  1. Every neighbor is weighted the same way (up to a fixed, node-degree-
     based normalization factor) - the model can't learn that some Hi-C
     contacts matter far more than others.
  2. No residual connections - each node's own signal (its DNABERT-2
     embedding) gets diluted by repeated neighborhood averaging, with
     nothing preserving the original per-node representation.
  3. A missing nonlinearity between layers 2 and 3 (F.relu is applied
     after gcn1 but never after gcn2) collapsed those two layers into one
     effective linear transform - less expressive than a real 3-layer
     network, on top of problems 1-2.

GATv2 (Brody et al., 2022 - "How Attentive Are Graph Attention Networks?",
a fix to the original GAT's provably static/rank-1 attention) lets each
node learn WHICH neighbors to weight more heavily, conditioned on both
nodes' current features - exactly the adaptive weighting plain GCN can't
express. Combined here with:

  - edge_dim=4: a 4-feature edge representation - [log1p(KR/SCALE-
    normalized contact count), observed/expected ratio, log1p(genomic
    distance in bp), is_self_loop] (see feature_extraction/prepare_hic_graph.py's
    compute_oe_edge_features) - fed into GATv2Conv's attention computation,
    so attention is informed by real Hi-C contact strength AND how
    unusual that strength is FOR ITS DISTANCE (the O/E ratio - Lieberman-
    Aiden et al., 2009 - separates real loops/TADs from the trivial
    "closer bins touch more" trend, which raw/KR counts alone don't), not
    just node content. Earlier version here used a single degree-
    normalized adjacency value as edge_dim=1 - wrong signal for attention,
    since that value conflates true contact strength with how connected a
    node is overall (a GCN-specific correction, not meaningful to GATv2's
    own attention-based aggregation) - see project history.
  - Residual connections around each GATv2 layer (He et al., 2016;
    DeepGCNs - Li et al., 2019, applies the same idea to GNNs
    specifically) - preserves each node's own DNABERT-2 signal instead of
    letting it wash out under repeated neighborhood averaging.
  - Jumping Knowledge (Xu et al., 2018) - concatenates every layer's
    representation (the pre-GAT projection plus both GAT layers' outputs)
    before the final readout, instead of reading out only the last layer,
    so the model can keep shallow (node-local) and deep (neighborhood-
    aggregated) information side by side rather than being forced to pick
    one.

Requires (Colab): !pip install torch_geometric

Input:  node_input  [N, 768]  (unchanged - same DNABERT-2 node features)
        edge_index  [2, E]    (see load_oe_edge_index below - reads
                               edge_features.npz, produced once by
                               feature_extraction/prepare_hic_graph.py's
                               compute_oe_edge_features)
        edge_attr   [E, 4]
        node_index  [B]       (unchanged)
Output: [B, 128] (unchanged - dimension-compatible with GatedFusion's
        graph_dim, so it's a drop-in replacement at the fusion boundary
        even though its forward signature differs from GCN_Structure's -
        see model/graph_branch.py's docstring for the [N, N] dense/sparse
        adjacency it expects instead)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# Lowered from an initial 256/4 after a real run OOM'd at ~95GB: GATv2Conv's
# attention computation materializes several [E, heads, out_channels]
# intermediate tensors per layer (retained for backward), and with E~2.7M
# edges even the post-distance-cap graph, 2 layers x 4 heads x 256 dims
# added up to more than the GPU's full 95GB. heads and out_channels both
# scale that tensor size roughly linearly - halving each here, combined
# with mixed precision in training/train_graph_gat.py's run_epoch, cuts
# peak memory by roughly 8x.
HIDDEN_DIM = 128
NUM_HEADS = 2
DROPOUT_PROB = 0.2


def load_oe_edge_index(edge_features_path) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads feature_extraction/prepare_hic_graph.py's compute_oe_edge_features
    output (edge_features.npz: rows, cols, features[E, 4]) into PyTorch
    Geometric's edge_index/edge_attr format. Self-loops are already
    present in this file (one per node, added explicitly by
    compute_oe_edge_features) - GATv2Structure is built with
    add_self_loops=False below so they aren't added a second time under a
    different (uninformative, all-zero-but-flag) value."""
    with np.load(edge_features_path) as data:
        rows = data["rows"]
        cols = data["cols"]
        features = data["features"]
    edge_index = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
    edge_attr = torch.from_numpy(features.astype(np.float32))
    return edge_index, edge_attr


class GATv2Structure(nn.Module):
    """
    GATv2-based Hi-C graph branch with residual connections and Jumping
    Knowledge readout - see module docstring for what this replaces
    GCN_Structure for and why.

    Inputs:
        node_input:
            DNABERT-2 node features [N, 768].

        edge_index:
            Graph connectivity [2, E] (long tensor).

        edge_attr:
            4-feature Hi-C edge representation [E, 4] (float tensor) -
            [log1p(count), observed/expected, log1p(distance), is_self_loop],
            see feature_extraction/prepare_hic_graph.py's compute_oe_edge_features.

        node_index:
            Graph node index per sample [B] (long tensor).

    Output:
        Selected graph representations [B, 128].
    """

    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(768, HIDDEN_DIM)

        # concat=False: average the attention heads back to HIDDEN_DIM
        # instead of concatenating them (which would multiply the width by
        # NUM_HEADS) - keeps every layer's width fixed at HIDDEN_DIM, which
        # the residual add below requires.
        self.gat1 = GATv2Conv(
            HIDDEN_DIM, HIDDEN_DIM, heads=NUM_HEADS, concat=False,
            edge_dim=4, dropout=DROPOUT_PROB, add_self_loops=False,
        )
        self.gat2 = GATv2Conv(
            HIDDEN_DIM, HIDDEN_DIM, heads=NUM_HEADS, concat=False,
            edge_dim=4, dropout=DROPOUT_PROB, add_self_loops=False,
        )

        self.dropout = nn.Dropout(p=DROPOUT_PROB)

        # Jumping Knowledge: concatenate the pre-GAT projection + both GAT
        # layers' outputs (3 x HIDDEN_DIM) instead of reading out only the
        # last layer.
        self.readout = nn.Linear(HIDDEN_DIM * 3, 128)

    def forward(
        self,
        node_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_index: torch.Tensor,
    ) -> torch.Tensor:
        h0 = F.elu(self.input_projection(node_input))  # [N, 768] -> [N, HIDDEN_DIM]

        h1 = F.elu(self.gat1(h0, edge_index, edge_attr)) + h0
        h1 = self.dropout(h1)

        h2 = F.elu(self.gat2(h1, edge_index, edge_attr)) + h1
        h2 = self.dropout(h2)

        jumping_knowledge = torch.cat([h0, h1, h2], dim=1)  # [N, HIDDEN_DIM * 3]
        structure_output = self.readout(jumping_knowledge)  # [N, 128]

        return structure_output[node_index]
