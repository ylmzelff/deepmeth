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

        index_input:
            One-hot node selection matrix [B, N]

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
        index_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_input:
                Node features [N, 768].

            adj_input:
                Normalized sparse adjacency [N, N].

            index_input:
                One-hot node selection matrix [B, N].

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

        # Original ncVarPred node-selection operation:
        #
        # [B, N] x [N, 128] -> [B, 128]
        structure_output = torch.matmul(
            index_input,
            structure_output,
        )

        return structure_output



    