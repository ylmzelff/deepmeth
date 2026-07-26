import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Additive attention pooling (Lin et al., 2017-style: a small 2-layer
    scoring MLP with tanh, softmax over the sequence dimension, weighted
    sum). Replaces flatten + a large FC: instead of treating all 36
    BiLSTM time steps as one fixed concatenated vector, the model learns
    a per-sample weight for each time step - i.e. which motif regions
    actually matter for this CpG - before collapsing to a fixed-size
    vector. The weights are also interpretable (see last_attention_weights
    on DanQ_Sequence).

    Input:  [B, L, D]
    Output: [B, D]
    """

    def __init__(self, input_dim: int, attention_hidden_dim: int):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(sequence)  # [B, L, 1]
        weights = torch.softmax(scores, dim=1)  # softmax over L (positions), not features
        pooled = (sequence * weights).sum(dim=1)  # [B, D]
        return pooled, weights.squeeze(-1)  # weights: [B, L], for interpretability


NUM_CNN_POSITIONS = 36  # (501 - 26 + 1) // 13, fixed by SEQUENCE_LENGTH=501


class DanQ_Sequence(nn.Module):
    """
    DanQ-based 1D DNA sequence branch adapted from ncVarPred.

    Input:
        [batch_size, 4, 501]

    Output:
        [batch_size, 925]

    use_self_attention=False (default) reproduces the original ncVarPred
    DanQ path bit-for-bit: Conv1d -> MaxPool -> BiLSTM -> attention pooling.

    use_self_attention=True replaces the BiLSTM with a lightweight
    Transformer encoder (self-attention) over the same 36 CNN positions -
    cheap (36x36 attention, negligible next to the graph branch's cost),
    and lets positions attend directly to any other position instead of
    only through a sequential BiLSTM bottleneck. Since self-attention has
    no inherent notion of order, a learned positional embedding is added
    first. This is an ablation toggle: both paths are kept so the two can
    be trained and compared under the exact same script.
    """

    def __init__(self, use_self_attention: bool = False):
        super(DanQ_Sequence, self).__init__()

        self.use_self_attention = use_self_attention

        # Original ncVarPred DanQ convolution.
        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=320,
            kernel_size=26,
        )

        # Original ncVarPred pooling layer.
        self.maxpool = nn.MaxPool1d(
            kernel_size=13,
            stride=13,
        )

        self.drop1 = nn.Dropout(
            p=0.2
        )

        self.drop2 = nn.Dropout(
            p=0.5
        )

        if self.use_self_attention:
            # Learned positional embedding: self-attention has no built-in
            # sense of position order, unlike the BiLSTM it replaces.
            self.positional_embedding = nn.Parameter(
                torch.zeros(NUM_CNN_POSITIONS, 320)
            )
            nn.init.normal_(self.positional_embedding, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=320,
                nhead=8,
                dim_feedforward=640,
                dropout=0.1,
                batch_first=True,
            )
            self.self_attention_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
            )

            sequence_encoder_output_dim = 320
        else:
            # Original ncVarPred bidirectional LSTM.
            self.bilstm = nn.LSTM(
                input_size=320,
                hidden_size=320,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

            sequence_encoder_output_dim = 640  # 320 forward + 320 backward

        # For a 501 bp sequence:
        #
        # Conv1d:
        # 501 -> 476
        #
        # MaxPool1d:
        # 476 -> 36
        #
        # BiLSTM: 320 forward + 320 backward = 640
        # Self-attention (if enabled): stays at 320, no directional doubling
        #
        # Attention pooling collapses the 36 time steps to one vector
        # (weighted by learned per-position importance) instead of
        # flattening all 36 positions into a fixed-size vector - same
        # final output width (925), far fewer parameters in this layer.
        self.attention_pool = AttentionPooling(
            input_dim=sequence_encoder_output_dim,
            attention_hidden_dim=128,
        )

        self.last_attention_weights: torch.Tensor | None = None

        self.fc1 = nn.Linear(
            sequence_encoder_output_dim,
            925,
        )

    def forward(
        self,
        seq_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_input:
                One-hot DNA sequences with shape
                [B, 4, 501].

        Returns:
            Sequence representations with shape
            [B, 925].
        """

        # [B, 4, 501] -> [B, 320, 476]
        seq_output = self.conv1(
            seq_input
        )

        seq_output = F.relu(
            seq_output
        )

        # [B, 320, 476] -> [B, 320, 36]
        seq_output = self.maxpool(
            seq_output
        )

        seq_output = self.drop1(
            seq_output
        )

        # Conv1d produces [B, channels, length].
        # batch_first=True LSTM expects [B, length, features].
        #
        # [B, 320, 36] -> [B, 36, 320]
        seq_output = seq_output.permute(
            0,
            2,
            1,
        )

        if self.use_self_attention:
            # [B, 36, 320] + positional embedding -> self-attention -> [B, 36, 320]
            seq_output = seq_output + self.positional_embedding.unsqueeze(0)
            seq_output = self.self_attention_encoder(seq_output)
        else:
            # [B, 36, 320] -> [B, 36, 640]
            seq_output, _ = self.bilstm(
                seq_output
            )

        seq_output = self.drop2(
            seq_output
        )

        # [B, 36, D] -> [B, D] (attention-weighted, not flattened)
        seq_output, attention_weights = self.attention_pool(
            seq_output
        )

        # Stored for interpretability (e.g. plotting which of the 36 motif
        # regions the model attended to) - not read anywhere in the forward
        # pass itself, so it never affects gradients/training.
        self.last_attention_weights = attention_weights.detach()

        # [B, 640] -> [B, 925]
        seq_output = self.fc1(
            seq_output
        )

        seq_output = F.relu(
            seq_output
        )

        return seq_output
