import torch
import torch.nn as nn

from config.project_config import (
    FOUNDATION_BRANCH_OUTPUT_DIM,
    PHYSICOCHEMICAL_CNN_OUTPUT_DIM,
    SEQUENCE_BRANCH_OUTPUT_DIM,
)
from model.sequence_branch import DanQ_Sequence
from model.foundation_branch import DeepMethFoundationBranch
from model.physicochemical_branch import CNNNet_PhyChemDi
from model.fusion import GatedFusion


class DeepMethModel(nn.Module):
    
    def __init__(
        self,
        physchem_dropout_prob: float,
        fusion_projected_dim: int,
        fusion_hidden_dim: int,
        fusion_dropout_prob: float,
        use_physchem_property_gate: bool = False,
        foundation_dropout_prob: float = 0.2,
    ):
        super(DeepMethModel, self).__init__()

        # ncVarPred DanQ-based sequence branch - see model/sequence_branch.py.
        self.sequence_branch = DanQ_Sequence()

        # Frozen-DNABERT-2-token-embeddings branch - see model/foundation_branch.py.
        self.foundation_branch = DeepMethFoundationBranch(dropout_prob=foundation_dropout_prob)

        # Yeast Promoter physicochemical CNN branch. use_physchem_property_gate
        # is an independent toggle - see model/physicochemical_branch.py.
        # Defaults False (unchanged).
        self.physchem_branch = CNNNet_PhyChemDi(
            dropout_prob=physchem_dropout_prob,
            use_property_gate=use_physchem_property_gate,
        )

        # Gated fusion head instead of concatenation + one linear layer.
        self.fusion = GatedFusion(
            sequence_dim=SEQUENCE_BRANCH_OUTPUT_DIM,
            foundation_dim=FOUNDATION_BRANCH_OUTPUT_DIM,
            physchem_dim=PHYSICOCHEMICAL_CNN_OUTPUT_DIM,
            projected_dim=fusion_projected_dim,
            hidden_dim=fusion_hidden_dim,
            dropout_prob=fusion_dropout_prob,
        )

        

    def forward(
        self,
        seq_input: torch.Tensor,
        foundation_token_embeddings: torch.Tensor,
        foundation_attention_mask: torch.Tensor,
        physchem_input: torch.Tensor,
    ) -> torch.Tensor:
        

        # [B, 4, 501] -> [B, 925]
        seq_output = self.sequence_branch(seq_input)

        # [B, T, 768], [B, T] -> [B, 256]
        foundation_output = self.foundation_branch(
            foundation_token_embeddings,
            foundation_attention_mask,
        )

        # [B, 12, 500] -> [B, 480]
        physchem_output = self.physchem_branch(physchem_input)

        # Gated fusion: [B, 925] + [B, 256] + [B, 480] -> [B, 1]
        output = self.fusion(
            seq_output,
            foundation_output,
            physchem_output,
        )

        return output
