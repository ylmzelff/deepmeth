import torch
import torch.nn as nn

from model.sequence_branch import DanQ_Sequence
from model.graph_branch_gat import GATv2Structure
from model.graph_readout import CpGAwareGraphReadout
from model.physicochemical_branch import CNNNet_PhyChemDi
from model.fusion import GatedFusion


class DeepMethModel(nn.Module):
    """
    Three-branch DeepMeth model.

    Branch 1:
        One-hot DNA sequence
        [B, 4, 501] -> [B, 925]

    Branch 2:
        DNABERT-2 node features + Hi-C graph, via GATv2Structure (replaces
        the original GCN_Structure - see model/graph_branch_gat.py's module
        docstring for why), then conditioned on the target CpG's own
        sequence embedding via CpGAwareGraphReadout (see
        model/graph_readout.py's module docstring) so that CpGs sharing a
        Hi-C node no longer receive an identical graph representation - the
        one change able to exceed the node-index-only majority-vote oracle
        ceiling (project history).
        [N, NODE_FEATURE_DIM], edge_index [2, E], edge_attr [E, 4] -> [B, 128]

    Branch 3:
        Dinucleotide physicochemical matrix
        [B, 12, 500] -> [B, 480]

    Fusion:
        Gated fusion (see model/fusion.py) - each branch projected to a
        shared dimension, combined with a learned per-feature gate over
        the 3 modalities, then a small MLP head. Replaces the original
        ncVarPred "concatenate + one linear layer" fusion; the sequence and
        physicochemical branches themselves are unchanged.

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
        use_sequence_multiscale_cnn: bool = False,
        use_physchem_property_gate: bool = False,
        graph_readout_dropout_prob: float = 0.2,
    ):
        super(DeepMethModel, self).__init__()

        # ncVarPred DanQ-based sequence branch. use_sequence_self_attention
        # swaps the BiLSTM for a lightweight Transformer encoder;
        # use_sequence_multiscale_cnn swaps the single kernel_size=26 conv
        # for parallel multi-scale conv branches - see
        # model/sequence_branch.py. Both default False, reproducing the
        # original DanQ path unchanged.
        self.sequence_branch = DanQ_Sequence(
            use_self_attention=use_sequence_self_attention,
            use_multiscale_cnn=use_sequence_multiscale_cnn,
        )

        # GATv2-based structure branch (see model/graph_branch_gat.py) +
        # CpG-aware readout (see model/graph_readout.py) - together replace
        # the original GCN_Structure.
        self.structure_branch = GATv2Structure()
        self.graph_readout = CpGAwareGraphReadout(
            graph_dim=128,
            sequence_dim=925,
            dropout_prob=graph_readout_dropout_prob,
        )

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
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_index: torch.Tensor,
        physchem_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_input:
                One-hot DNA sequences [B, 4, 501].

            node_input:
                DNABERT-2 + positional/confidence node features
                [N, NODE_FEATURE_DIM] (see model/graph_branch_gat.py).

            edge_index:
                Hi-C graph connectivity [2, E] (long tensor).

            edge_attr:
                4-feature Hi-C edge representation [E, 4] (float tensor) -
                see feature_extraction/prepare_hic_graph.py's
                compute_oe_edge_features.

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

        # [N, NODE_FEATURE_DIM], [2, E], [E, 4], [B] -> [B, 128]
        raw_structure_output = self.structure_branch(
            node_input,
            edge_index,
            edge_attr,
            node_index,
        )
        # [B, 128], [B, 925] -> [B, 128], conditioned on each sample's own
        # sequence embedding - see model/graph_readout.py.
        structure_output = self.graph_readout(raw_structure_output, seq_output)

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
