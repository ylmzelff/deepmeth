import torch
import torch.nn as nn

from model.sequence_branch import DanQ_Sequence
from model.graph_branch import GCN_Structure
from model.physicochemical_branch import CNNNet_PhyChemDi
from model.fusion import GatedFusion


class DeepMethModel(nn.Module):
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
        Gated fusion (see model/fusion.py) - each branch projected to a
        shared dimension, combined with a learned per-feature gate over
        the 3 modalities, then a small MLP head. Replaces the original
        ncVarPred "concatenate + one linear layer" fusion; the three
        branches themselves are unchanged.

    Output:
        Raw methylation logit [B, 1] (unnormalized - pass through
        torch.sigmoid to get a probability, or use directly with
        nn.BCEWithLogitsLoss for training)
    """

    def __init__(
        self,
        physchem_dropout_prob: float,
        fusion_projected_dim: int,
        fusion_hidden_dim: int,
        fusion_dropout_prob: float,
        use_sequence_self_attention: bool = False,
        use_physchem_property_gate: bool = False,
    ):
        super(DeepMethModel, self).__init__()

        # ncVarPred DanQ-based sequence branch. use_sequence_self_attention
        # swaps the BiLSTM for a lightweight Transformer encoder over the
        # same 36 CNN positions - see model/sequence_branch.py. Default
        # False reproduces the original DanQ path unchanged.
        self.sequence_branch = DanQ_Sequence(
            use_self_attention=use_sequence_self_attention
        )

        # ncVarPred GCN-based structure branch.
        self.structure_branch = GCN_Structure()

        # Yeast Promoter physicochemical CNN branch. use_physchem_property_gate
        # is an independent toggle - see model/physicochemical_branch.py.
        # Defaults False (unchanged).
        self.physchem_branch = CNNNet_PhyChemDi(
            dropout_prob=physchem_dropout_prob,
            use_property_gate=use_physchem_property_gate,
        )

        # Gated fusion head instead of concatenation + one linear layer.
        #
        # Sequence:        925
        # Structure:       128
        # Physicochemical: 480
        self.fusion = GatedFusion(
            sequence_dim=925,
            graph_dim=128,
            physchem_dim=480,
            projected_dim=fusion_projected_dim,
            hidden_dim=fusion_hidden_dim,
            dropout_prob=fusion_dropout_prob,
        )

        # No sigmoid here - the model returns a raw logit. Training uses
        # nn.BCEWithLogitsLoss(pos_weight=...) directly on that logit
        # (numerically stabler than a separate sigmoid + BCELoss, and lets
        # the class-imbalance pos_weight be applied); torch.sigmoid is
        # applied afterward only where an actual probability is needed
        # (metrics, thresholding at inference).

    def forward(
        self,
        seq_input: torch.Tensor,
        node_input: torch.Tensor,
        adj_input: torch.Tensor,
        node_index: torch.Tensor,
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

            node_index:
                Graph node index per sample [B] (long tensor).

            physchem_input:
                Physicochemical feature matrix [B, 12, 500].

        Returns:
            Raw methylation logits [B, 1].
        """

        # [B, 4, 501] -> [B, 925]
        seq_output = self.sequence_branch(
            seq_input
        )

        # [N, 768], [N, N], [B] -> [B, 128]
        structure_output = self.structure_branch(
            node_input=node_input,
            adj_input=adj_input,
            node_index=node_index,
        )

        # [B, 12, 500] -> [B, 480]
        physchem_output = self.physchem_branch(
            physchem_input
        )

        # Gated fusion: [B, 925] + [B, 128] + [B, 480] -> [B, 1]
        output = self.fusion(
            seq_output,
            structure_output,
            physchem_output,
        )

        return output


