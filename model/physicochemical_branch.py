from pathlib import Path

import torch
import torch.nn as nn

from feature_extraction.physicochemical_features import (
    convertSampleToPhyChemVector_Di,
    load_physicochemical_properties_di,
)


MAXPOOL1D_KERNEL_SIZE = 2
CONV1D_KERNEL_SIZE = 3

CONV1D_FEATURE_SIZE_BLOCK1 = 32
CONV1D_FEATURE_SIZE_BLOCK2 = 16
CONV1D_FEATURE_SIZE_BLOCK3 = 8


class CNNNet_PhyChemDi(nn.Module):
    """
    CNN branch for dinucleotide physicochemical features.

    Input:
        [batch_size, 12, 500]

    Output:
        [batch_size, 480]
    """

    def __init__(self, dropout_prob: float):
        super(CNNNet_PhyChemDi, self).__init__()

        self.dropout_prob = dropout_prob

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=12,
                out_channels=CONV1D_FEATURE_SIZE_BLOCK1,
                kernel_size=CONV1D_KERNEL_SIZE,
            ),
            nn.BatchNorm1d(
                CONV1D_FEATURE_SIZE_BLOCK1
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool1d(
                MAXPOOL1D_KERNEL_SIZE
            )
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=CONV1D_FEATURE_SIZE_BLOCK1,
                out_channels=CONV1D_FEATURE_SIZE_BLOCK2,
                kernel_size=CONV1D_KERNEL_SIZE,
            ),
            nn.BatchNorm1d(
                CONV1D_FEATURE_SIZE_BLOCK2
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
        )

        self.pool2 = nn.Sequential(
            nn.MaxPool1d(
                MAXPOOL1D_KERNEL_SIZE
            )
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=CONV1D_FEATURE_SIZE_BLOCK2,
                out_channels=CONV1D_FEATURE_SIZE_BLOCK3,
                kernel_size=CONV1D_KERNEL_SIZE,
            ),
            nn.BatchNorm1d(
                CONV1D_FEATURE_SIZE_BLOCK3
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
        )

        self.pool3 = nn.Sequential(
            nn.MaxPool1d(
                MAXPOOL1D_KERNEL_SIZE
            )
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:
                Physicochemical feature tensor
                with shape [B, 12, 500].

        Returns:
            Flattened CNN features with
            shape [B, 480].
        """

        batch_size = inputs.size(0)

        output = self.pool1(
            self.conv1(inputs)
        )

        output = self.pool2(
            self.conv2(output)
        )

        output = self.pool3(
            self.conv3(output)
        )

        output = output.view(
            batch_size,
            -1,
        )

        return output



    