import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):

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


class DanQ_Sequence(nn.Module):
    """
    DanQ-based 1D DNA sequence branch adapted from ncVarPred.

    Input:
        [batch_size, 4, 501]

    Output:
        [batch_size, 925]

    Reproduces the original ncVarPred DanQ path bit-for-bit: Conv1d ->
    MaxPool -> BiLSTM -> attention pooling.
    """

    def __init__(self):
        super(DanQ_Sequence, self).__init__()

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

        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)

        # Original ncVarPred bidirectional LSTM.
        self.bilstm = nn.LSTM(
            input_size=320,
            hidden_size=320,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        sequence_encoder_output_dim = 640
        self.attention_pool = AttentionPooling(
            input_dim=sequence_encoder_output_dim,
            attention_hidden_dim=128,
        )

        self.last_attention_weights: torch.Tensor | None = None

        self.fc1 = nn.Linear(sequence_encoder_output_dim, 925)

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
        seq_output = F.relu(self.conv1(seq_input))

        # [B, 320, 476] -> [B, 320, 36]
        seq_output = self.drop1(self.maxpool(seq_output))

        # Conv1d produces [B, channels, length]; batch_first=True LSTM
        # expects [B, length, features].
        # [B, 320, 36] -> [B, 36, 320]
        seq_output = seq_output.permute(0, 2, 1)

        # [B, 36, 320] -> [B, 36, 640]
        seq_output, _ = self.bilstm(seq_output)
        seq_output = self.drop2(seq_output)

        # [B, 36, D] -> [B, D] (attention-weighted, not flattened)
        seq_output, attention_weights = self.attention_pool(seq_output)

        # Stored for interpretability (e.g. plotting which of the 36 motif
        # regions the model attended to) - not read anywhere in the forward
        # pass itself, so it never affects gradients/training.
        self.last_attention_weights = attention_weights.detach()

        # [B, 640] -> [B, 925]
        return F.relu(self.fc1(seq_output))
