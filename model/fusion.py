import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        foundation_dim: int,
        physchem_dim: int,
        projected_dim: int,
        hidden_dim: int,
        dropout_prob: float,
    ):
        super(GatedFusion, self).__init__()

        self.projected_dim = projected_dim

        self.sequence_projection = nn.Linear(sequence_dim, projected_dim)
        self.foundation_projection = nn.Linear(foundation_dim, projected_dim)
        self.physchem_projection = nn.Linear(physchem_dim, projected_dim)
        self.gate_network = nn.Linear(
            sequence_dim + foundation_dim + physchem_dim,
            3 * projected_dim,
        )

        self.head = nn.Sequential(
            nn.Linear(projected_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        seq_output: torch.Tensor,
        foundation_output: torch.Tensor,
        physchem_output: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = seq_output.size(0)

        raw_concatenated = torch.cat([seq_output, foundation_output, physchem_output], dim=1)

        # [B, 3 * projected_dim] -> [B, 3, projected_dim], softmax across the 3 modalities.
        gate_logits = self.gate_network(raw_concatenated).view(batch_size, 3, self.projected_dim)
        gates = torch.softmax(gate_logits, dim=1)

        sequence_projected = torch.tanh(self.sequence_projection(seq_output))
        foundation_projected = torch.tanh(self.foundation_projection(foundation_output))
        physchem_projected = torch.tanh(self.physchem_projection(physchem_output))

        fused = (
            gates[:, 0, :] * sequence_projected
            + gates[:, 1, :] * foundation_projected
            + gates[:, 2, :] * physchem_projected
        )

        return self.head(fused)
