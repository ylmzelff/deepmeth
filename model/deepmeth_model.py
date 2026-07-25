import torch
import torch.nn as nn

from model.sequence_branch import DanQ_Sequence
from model.graph_branch import GCN_Structure
from model.physicochemical_branch import CNNNet_PhyChemDi


class DeepMethConcatenation(nn.Module):
    """
    Three-branch DeepMeth model.

    Branch 1:
        One-hot DNA sequence
        [B, 4, 501] -> [B, 925]

    Branch 2:
        DNABERT node features and Hi-C graph
        [N, 768] -> [B, 128]

    Branch 3:
        Dinucleotide physicochemical matrix
        [B, 12, 500] -> [B, 480]

    Fusion:
        925 + 128 + 480 = 1533

    Output:
        Binary methylation probability [B, 1]
    """

    def __init__(
        self,
        physchem_dropout_prob: float,
    ):
        super(DeepMethConcatenation, self).__init__()

        # ncVarPred DanQ-based sequence branch.
        self.sequence_branch = DanQ_Sequence()

        # ncVarPred GCN-based structure branch.
        self.structure_branch = GCN_Structure()

        # Yeast Promoter physicochemical CNN branch.
        self.physchem_branch = CNNNet_PhyChemDi(
            dropout_prob=physchem_dropout_prob
        )

        # Original ncVarPred model concatenates branch outputs
        # and applies one fully connected prediction layer.
        #
        # Sequence:        925
        # Structure:       128
        # Physicochemical: 480
        #
        # Total:          1533
        self.fc2 = nn.Linear(
            925 + 128 + 480,
            1,
        )

        # The original ncVarPred model applies sigmoid
        # after the final fully connected layer.
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        seq_input: torch.Tensor,
        node_input: torch.Tensor,
        adj_input: torch.Tensor,
        index_input: torch.Tensor,
        physchem_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_input:
                One-hot DNA sequences [B, 4, 501].

            node_input:
                DNABERT node features [N, 768].

            adj_input:
                Normalized Hi-C adjacency matrix [N, N].

            index_input:
                One-hot graph node selection matrix [B, N].

            physchem_input:
                Physicochemical feature matrix [B, 12, 500].

        Returns:
            Methylation probabilities [B, 1].
        """

        # [B, 4, 501] -> [B, 925]
        seq_output = self.sequence_branch(
            seq_input
        )

        # [N, 768], [N, N], [B, N] -> [B, 128]
        structure_output = self.structure_branch(
            node_input=node_input,
            adj_input=adj_input,
            index_input=index_input,
        )

        # [B, 12, 500] -> [B, 480]
        physchem_output = self.physchem_branch(
            physchem_input
        )

        # Original concatenation operation extended
        # from two branches to three branches.
        #
        # [B, 925] + [B, 128] + [B, 480]
        # -> [B, 1533]
        concatenated_output = torch.cat(
            (
                seq_output,
                structure_output,
                physchem_output,
            ),
            dim=1,
        )

        # [B, 1533] -> [B, 1]
        output = self.fc2(
            concatenated_output
        )

        # Binary methylation probability.
        output = self.sigmoid(
            output
        )

        return output


