

from __future__ import annotations

import torch
import torch.nn as nn


class CpGAwareGraphReadout(nn.Module):

    def __init__(self, graph_dim: int, sequence_dim: int, dropout_prob: float = 0.2):
        super().__init__()

        self.sequence_projection = nn.Linear(sequence_dim, graph_dim)
        self.gate = nn.Sequential(
            nn.Linear(graph_dim * 2, graph_dim),
            nn.Sigmoid(),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(graph_dim * 2, graph_dim),
            nn.ELU(),
            nn.Dropout(dropout_prob),
        )

    def forward(self, graph_embedding: torch.Tensor, sequence_embedding: torch.Tensor) -> torch.Tensor:
        projected_sequence = self.sequence_projection(sequence_embedding)  

        gate = self.gate(torch.cat([graph_embedding, projected_sequence], dim=1))
        conditioned_graph = gate * graph_embedding

        return self.output_projection(torch.cat([conditioned_graph, projected_sequence], dim=1))
