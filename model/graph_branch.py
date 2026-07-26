import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import GraphConvolution


class GCN_Structure(nn.Module):
    """
    ncVarPred-based 3D genome graph branch.

    Inputs:
        node_input:
            DNABERT node features [N, 768]

        adj_input:
            Normalized Hi-C adjacency matrix [N, N]

        node_index:
            Graph node index per sample [B] (long tensor)

    Output:
        Selected graph representations [B, 128]
    """

    def __init__(self):
        super(GCN_Structure, self).__init__()

        # Original ncVarPred GCN dimensions.
        self.gcn1 = GraphConvolution(
            in_features=768,
            out_features=1000,
        )

        self.gcn2 = GraphConvolution(
            in_features=1000,
            out_features=400,
        )

        self.gcn3 = GraphConvolution(
            in_features=400,
            out_features=128,
        )

        # Same dropout probability used in ncVarPred.
        self.drop1 = nn.Dropout(
            p=0.2
        )

    def forward(
        self,
        node_input: torch.Tensor,
        adj_input: torch.Tensor,
        node_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_input:
                Node features [N, 768].

            adj_input:
                Normalized sparse adjacency [N, N].

            node_index:
                Graph node index per sample [B] (long tensor).

        Returns:
            Graph features [B, 128].
        """

        # [N, 768] -> [N, 1000]
        structure_output = F.relu(
            self.gcn1(
                node_input,
                adj_input,
            )
        )

        structure_output = self.drop1(
            structure_output
        )

        # [N, 1000] -> [N, 400]
        structure_output = self.gcn2(
            structure_output,
            adj_input,
        )

        structure_output = self.drop1(
            structure_output
        )

        # [N, 400] -> [N, 128]
        structure_output = self.gcn3(
            structure_output,
            adj_input,
        )

        # Row-gather equivalent of the original ncVarPred one-hot
        # node-selection matmul ([B, N] x [N, 128] -> [B, 128]), but O(B)
        # instead of O(B x N): each sample's node is a single row lookup,
        # not a full-width dot product against a mostly-zero one-hot row.
        # Identical result, far less compute and memory traffic.
        structure_output = structure_output[node_index]

        return structure_output



    