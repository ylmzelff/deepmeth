from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from config.project_config import DNABERT_HIDDEN_SIZE, GRAPH_BRANCH_OUTPUT_DIM, INCLUDED_CHROMOSOMES

EXTRA_NODE_FEATURE_DIM = len(INCLUDED_CHROMOSOMES) + 2
NODE_FEATURE_DIM = DNABERT_HIDDEN_SIZE + EXTRA_NODE_FEATURE_DIM
HIDDEN_DIM = 128
NUM_HEADS = 2
DROPOUT_PROB = 0.2


def load_oe_edge_index(edge_features_path) -> tuple[torch.Tensor, torch.Tensor]:
   
    with np.load(edge_features_path) as data:
        rows = data["rows"]
        cols = data["cols"]
        features = data["features"]
    edge_index = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
    edge_attr = torch.from_numpy(features.astype(np.float32))
    return edge_index, edge_attr


class GATv2Structure(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(NODE_FEATURE_DIM, HIDDEN_DIM)

        
        self.gat1 = GATv2Conv(
            HIDDEN_DIM, HIDDEN_DIM, heads=NUM_HEADS, concat=False,
            edge_dim=4, dropout=DROPOUT_PROB, add_self_loops=False,
        )
        self.gat2 = GATv2Conv(
            HIDDEN_DIM, HIDDEN_DIM, heads=NUM_HEADS, concat=False,
            edge_dim=4, dropout=DROPOUT_PROB, add_self_loops=False,
        )

        self.dropout = nn.Dropout(p=DROPOUT_PROB)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.readout = nn.Linear(HIDDEN_DIM * 3, GRAPH_BRANCH_OUTPUT_DIM)

    def forward(
        self,
        node_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_index: torch.Tensor,
    ) -> torch.Tensor:
        h0 = F.elu(self.input_projection(node_input))  # [N, NODE_FEATURE_DIM] -> [N, HIDDEN_DIM]

        message1 = self.gat1(h0, edge_index, edge_attr)
        h1 = self.norm1(h0 + self.dropout(F.elu(message1)))

        message2 = self.gat2(h1, edge_index, edge_attr)
        h2 = self.norm2(h1 + self.dropout(F.elu(message2)))

        jumping_knowledge = torch.cat([h0, h1, h2], dim=1)  # [N, HIDDEN_DIM * 3]
        structure_output = self.readout(jumping_knowledge)  # [N, 128]

        return structure_output[node_index]
