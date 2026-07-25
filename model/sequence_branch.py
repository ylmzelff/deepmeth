import torch
import torch.nn as nn
import torch.nn.functional as F


class DanQ_Sequence(nn.Module):
    """
    DanQ-based 1D DNA sequence branch adapted from ncVarPred.

    Input:
        [batch_size, 4, 501]

    Output:
        [batch_size, 925]
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

        self.drop1 = nn.Dropout(
            p=0.2
        )

        # Original ncVarPred bidirectional LSTM.
        self.bilstm = nn.LSTM(
            input_size=320,
            hidden_size=320,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.drop2 = nn.Dropout(
            p=0.5
        )

        # For a 501 bp sequence:
        #
        # Conv1d:
        # 501 -> 476
        #
        # MaxPool1d:
        # 476 -> 36
        #
        # BiLSTM:
        # 320 forward + 320 backward = 640
        #
        # Flatten:
        # 36 * 640 = 23040
        self.fc1 = nn.Linear(
            36 * 640,
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

        # [B, 36, 320] -> [B, 36, 640]
        seq_output, _ = self.bilstm(
            seq_output
        )

        # [B, 36, 640] -> [B, 23040]
        seq_output = (
            seq_output
            .contiguous()
            .view(
                seq_output.size(0),
                36 * 640,
            )
        )

        seq_output = self.drop2(
            seq_output
        )

        # [B, 23040] -> [B, 925]
        seq_output = self.fc1(
            seq_output
        )

        seq_output = F.relu(
            seq_output
        )

        return seq_output
