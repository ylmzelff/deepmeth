import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated fusion head, replacing plain concatenation + one linear layer.

    Each branch's output is projected to a shared dimension (tanh), and a
    per-feature gate over the 3 modalities (softmax-normalized, so the
    three gates for a given feature sum to 1) is computed from the raw
    concatenated branch outputs. The gated projections are summed and
    passed through a small MLP head.

    This lets the model learn, per feature, how much to weight each
    modality instead of always trusting a fixed linear combination of
    all 1533 raw dimensions - standard practice for fusing branches of
    very different dimensionality and character (one-hot sequence,
    graph embedding, physicochemical CNN features).

    Input:
        seq_output:       [B, sequence_dim]
        structure_output: [B, graph_dim]
        physchem_output:  [B, physchem_dim]

    Output:
        Raw logit [B, 1].
    """

    def __init__(
        self,
        sequence_dim: int,
        graph_dim: int,
        physchem_dim: int,
        projected_dim: int,
        hidden_dim: int,
        dropout_prob: float,
    ):
        super(GatedFusion, self).__init__()

        self.projected_dim = projected_dim

        self.sequence_projection = nn.Linear(sequence_dim, projected_dim)
        self.graph_projection = nn.Linear(graph_dim, projected_dim)
        self.physchem_projection = nn.Linear(physchem_dim, projected_dim)

        # Per-feature gate logits for all 3 modalities at once: [B, 3 * projected_dim] -> [B, 3, projected_dim].
        self.gate_network = nn.Linear(
            sequence_dim + graph_dim + physchem_dim,
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
        structure_output: torch.Tensor,
        physchem_output: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = seq_output.size(0)

        raw_concatenated = torch.cat([seq_output, structure_output, physchem_output], dim=1)

        # [B, 3 * projected_dim] -> [B, 3, projected_dim], softmax across the 3 modalities.
        gate_logits = self.gate_network(raw_concatenated).view(batch_size, 3, self.projected_dim)
        gates = torch.softmax(gate_logits, dim=1)

        sequence_projected = torch.tanh(self.sequence_projection(seq_output))
        graph_projected = torch.tanh(self.graph_projection(structure_output))
        physchem_projected = torch.tanh(self.physchem_projection(physchem_output))

        fused = (
            gates[:, 0, :] * sequence_projected
            + gates[:, 1, :] * graph_projected
            + gates[:, 2, :] * physchem_projected
        )

        return self.head(fused)
