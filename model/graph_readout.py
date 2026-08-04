"""
CpG-aware conditional graph readout.

Motivation: GATv2Structure's forward pass ends with
`structure_output[node_index]` (see model/graph_branch_gat.py) - every CpG
mapped to the same Hi-C node receives the IDENTICAL graph embedding. This
is a hard ceiling, not a training/architecture weakness: on GM12878's 25kb
graph, feature_extraction/analyze_graph_node_label_purity.py's majority-
vote-per-node oracle (the best any node-index-only model could ever
achieve) is MCC~0.69, and three architecturally different graph branches
tried so far - the original GCN, GATv2Structure, and GATv2Structure with
richer O/E-based edge features (see project history) - all converged to
roughly 83-85% of that ceiling. Better edge features and node features
only push the branch closer to 0.69; none of them can cross it, because
none of them let the branch's output vary WITHIN a node.

CpGAwareGraphReadout crosses that ceiling by design: it takes the shared
node-level graph embedding and the CpG's OWN local sequence embedding
(already computed by the sequence branch in the same forward pass, not a
new encoder) and produces a per-CpG-specific graph representation via a
learned, per-feature gate - "how much of this node's 3D context applies to
THIS particular CpG, given what its own sequence looks like."

Trade-off, stated plainly: this couples the graph branch's final output to
the sequence branch's output, breaking the project's established "three
independent branches, each separately pretrainable and warm-startable"
design (see training/train.py, training/train_graph_gat.py). GATv2Structure
itself is unaffected - still trainable/warm-startable standalone via
training/train_graph_gat.py, unchanged. Only this readout layer, which sits
between GATv2Structure and GatedFusion, requires the full model's sequence
output and is trained from a random initialization as part of end-to-end
fine-tuning - it has no standalone pretraining stage of its own, by
necessity (there is no "CpG-specific target" to pretrain it against
independent of the sequence branch and the real downstream task).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CpGAwareGraphReadout(nn.Module):
    """
    Conditions a shared node-level graph embedding on the target CpG's own
    sequence embedding via a learned per-feature gate - see module
    docstring for the ceiling this is meant to cross and why it can't be
    pretrained standalone.

    Input:
        graph_embedding:
            GATv2Structure's per-sample output [B, graph_dim]
            (structure_output[node_index] - identical for every CpG
            sharing a node, before this module runs).

        sequence_embedding:
            The sequence branch's own per-CpG output [B, sequence_dim]
            (e.g. DanQ_Sequence's [B, 925]), computed in the same forward
            pass - reused, not a new encoder.

    Output:
        [B, graph_dim] - a CpG-specific graph representation, meant to
        replace the raw graph_embedding as GatedFusion's graph-branch
        input.
    """

    def __init__(self, graph_dim: int, sequence_dim: int, dropout_prob: float = 0.2):
        super().__init__()

        self.sequence_projection = nn.Linear(sequence_dim, graph_dim)

        # Per-feature gate (width graph_dim, not a single scalar): how much
        # of each dimension of the shared node embedding is relevant for
        # THIS specific CpG - conditioned on both the node's 3D context and
        # the CpG's own sequence, not a fixed, sample-independent blend.
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
        projected_sequence = self.sequence_projection(sequence_embedding)  # [B, sequence_dim] -> [B, graph_dim]

        gate = self.gate(torch.cat([graph_embedding, projected_sequence], dim=1))
        conditioned_graph = gate * graph_embedding

        return self.output_projection(torch.cat([conditioned_graph, projected_sequence], dim=1))
