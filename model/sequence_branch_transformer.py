"""
Transformer-based alternative to model/sequence_branch.py's DanQ_Sequence -
a genuinely deeper architecture change, not the earlier lightweight
"swap BiLSTM for one small self-attention layer" ablation (which lost,
val_mcc 0.64 vs BiLSTM's 0.65 - see project history). Kept as a separate
module, DanQ_Sequence itself untouched, so both can be trained and
compared under the same script (same pattern as model/cross_attention_fusion.py
being an alternative to model/fusion.py's GatedFusion).

Three ideas from the current DNA-transformer literature, combined (not
just one swapped-in technique):

  1. Multi-scale convolutional stem (Enformer, Tan & Shen 2023's
     Nucleic Transformer: conv layers first compress the raw one-hot
     sequence into motif-scale "tokens" before any attention runs, since
     attention over 501 raw bases is both wasteful and the wrong
     granularity - CpG-relevant motifs span multiple bases, not one).
     Three parallel Conv1d branches at different kernel sizes (9/15/25 bp)
     run over the same input and are concatenated, instead of DanQ's single
     kernel=26 branch - short and long motifs are captured at their own
     natural scale rather than forced through one fixed receptive field.

  2. ALiBi relative positional bias (Press, Smith & Lewis, 2021) inside
     self-attention, instead of an absolute learned positional embedding
     (which is what the existing self-attention ablation in
     sequence_branch.py uses). Rationale, specific to this task: what
     should matter for a regulatory motif's effect is its *spacing*
     relative to the CpG and to other motifs, not which absolute index
     (0-37) it happens to land on after conv downsampling - a purely
     relative signal is the better inductive bias here. Implemented from
     scratch (a custom multi-head attention) since nn.MultiheadAttention
     does not expose per-head additive bias injection.

  3. A real multi-layer encoder (default 4 layers, 8 heads) instead of a
     single attention layer, each block pre-LN residual self-attention +
     a feed-forward sublayer - standard modern Transformer practice
     (pre-LN improves training stability at this depth over the original
     post-LN design).

Attention pooling at the end is reused as-is from sequence_branch.py
(already validated there, not something this ablation touches).

Input:  [B, 4, 501] one-hot
Output: [B, 925] (kept identical to DanQ_Sequence's output width so this
        is a drop-in replacement for model/deepmeth_model.py's
        sequence_branch - GatedFusion's sequence_dim=925 does not need to
        change to use this).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sequence_branch import AttentionPooling

CONV_BRANCH_KERNEL_SIZES = (9, 15, 25)
CONV_BRANCH_OUT_CHANNELS = 128  # per branch; concatenated -> 3 * 128 = 384
STEM_POOL_KERNEL_SIZE = 13
STEM_POOL_STRIDE = 13


class MultiScaleConvStem(nn.Module):
    """
    Parallel Conv1d branches at different kernel sizes over the same
    one-hot input, concatenated along the channel dimension, then
    downsampled - the multi-scale analogue of DanQ_Sequence's single
    Conv1d(kernel=26).

    Input:  [B, 4, 501]
    Output: [B, sum(CONV_BRANCH_OUT_CHANNELS), L] where L = 38 for a
            501 bp input (matches DanQ_Sequence's 36-position bottleneck
            closely, incidental - not a hard requirement of anything
            downstream).
    """

    def __init__(self, dropout_prob: float):
        super().__init__()

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=4,
                        out_channels=CONV_BRANCH_OUT_CHANNELS,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # "same"-ish: keeps all branches length-501
                    ),
                    nn.BatchNorm1d(CONV_BRANCH_OUT_CHANNELS),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                )
                for kernel_size in CONV_BRANCH_KERNEL_SIZES
            ]
        )

        self.pool = nn.MaxPool1d(kernel_size=STEM_POOL_KERNEL_SIZE, stride=STEM_POOL_STRIDE)

    def forward(self, seq_input: torch.Tensor) -> torch.Tensor:
        # Each branch may be off-by-one in length depending on kernel
        # parity vs. the fixed padding=kernel_size//2 formula (odd
        # kernels are exact; guard against any mismatch by trimming to
        # the shortest branch before concatenating).
        branch_outputs = [branch(seq_input) for branch in self.branches]
        min_length = min(output.shape[-1] for output in branch_outputs)
        branch_outputs = [output[..., :min_length] for output in branch_outputs]

        concatenated = torch.cat(branch_outputs, dim=1)  # [B, 384, ~501]
        return self.pool(concatenated)  # [B, 384, ~38]


def _build_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Standard ALiBi slope schedule (Press et al., 2021): a geometric
    sequence so each head gets a different bias "steepness" - some heads
    end up nearly local, others attend more uniformly across the whole
    (short, ~38-position) sequence.
    """
    start_power = -8.0 / num_heads
    ratio = 2.0 ** start_power
    return torch.tensor([ratio ** (head + 1) for head in range(num_heads)], dtype=torch.float32)


class AlibiSelfAttention(nn.Module):
    """
    Multi-head self-attention with an ALiBi relative-position bias added
    to the attention logits, in place of any absolute positional
    embedding. Implemented directly (not via nn.MultiheadAttention, which
    has no hook for a per-head additive bias) - standard scaled
    dot-product attention plus one extra bias term.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_prob: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)

        self.register_buffer("alibi_slopes", _build_alibi_slopes(num_heads), persistent=False)

    def _alibi_bias(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        # [L, L]: relative distance |i - j|, negative so farther positions
        # are penalized (subtracted from the attention logit).
        relative_distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
        # [num_heads, L, L]
        return -self.alibi_slopes.to(device).view(-1, 1, 1) * relative_distance.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape

        query = self.query_projection(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_projection(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_projection(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        # query/key/value: [B, num_heads, L, head_dim]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores + self._alibi_bias(sequence_length, x.device)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, value)  # [B, num_heads, L, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)

        return self.output_projection(context)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN residual self-attention + pre-LN residual feed-forward - the
    modern (post-2020) Transformer block ordering, more stable to train
    at depth > 1-2 layers than the original post-LN design."""

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout_prob: float):
        super().__init__()

        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = AlibiSelfAttention(d_model, num_heads, dropout_prob)
        self.attention_dropout = nn.Dropout(dropout_prob)

        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feedforward_dim, d_model),
        )
        self.feedforward_dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_dropout(self.attention(self.attention_norm(x)))
        x = x + self.feedforward_dropout(self.feedforward(self.feedforward_norm(x)))
        return x


class TransformerSequence(nn.Module):
    """
    Multi-scale conv stem -> stack of ALiBi Transformer encoder blocks ->
    attention pooling -> linear. Drop-in alternative to
    model/sequence_branch.py's DanQ_Sequence (same [B,4,501] -> [B,925]
    interface).

    Input:  [B, 4, 501]
    Output: [B, 925]
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        conv_dropout_prob: float = 0.2,
        attention_dropout_prob: float = 0.15,
        pooling_dropout_prob: float = 0.3,
    ):
        super().__init__()

        self.stem = MultiScaleConvStem(dropout_prob=conv_dropout_prob)

        stem_output_channels = len(CONV_BRANCH_KERNEL_SIZES) * CONV_BRANCH_OUT_CHANNELS
        self.input_projection = nn.Linear(stem_output_channels, d_model)

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, attention_dropout_prob)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.pooling_dropout = nn.Dropout(pooling_dropout_prob)
        self.attention_pool = AttentionPooling(input_dim=d_model, attention_hidden_dim=128)
        self.last_attention_weights: torch.Tensor | None = None

        self.output_projection = nn.Linear(d_model, 925)

    def forward(self, seq_input: torch.Tensor) -> torch.Tensor:
        # [B, 4, 501] -> [B, 384, ~38]
        stem_output = self.stem(seq_input)

        # Conv1d gives [B, channels, length]; attention needs [B, length, channels].
        x = stem_output.permute(0, 2, 1)
        x = self.input_projection(x)  # [B, ~38, d_model]

        for block in self.encoder_blocks:
            x = block(x)
        x = self.final_norm(x)

        x = self.pooling_dropout(x)
        pooled, attention_weights = self.attention_pool(x)  # [B, d_model]
        self.last_attention_weights = attention_weights.detach()

        return F.relu(self.output_projection(pooled))  # [B, 925]
