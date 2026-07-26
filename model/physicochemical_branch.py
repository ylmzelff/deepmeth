import torch
import torch.nn as nn


MAXPOOL1D_KERNEL_SIZE = 2
CONV1D_KERNEL_SIZE = 3

CONV1D_FEATURE_SIZE_BLOCK1 = 32
CONV1D_FEATURE_SIZE_BLOCK2 = 16
CONV1D_FEATURE_SIZE_BLOCK3 = 8

NUM_PHYSICOCHEMICAL_PROPERTIES = 12


class CNNNet_PhyChemDi(nn.Module):
    """
    CNN branch for dinucleotide physicochemical features.

    Input:
        [batch_size, 12, 500]

    Output:
        [batch_size, 480]

    use_property_gate=True adds a learnable per-property scale (12 values,
    initialized to 1.0) applied to the raw input before any conv layer - a
    static/unconditional variant of FiLM (Perez et al., 2018). The learned
    weights are directly interpretable ("property X matters more than
    property Y for methylation").
    """

    def __init__(self, dropout_prob: float, use_property_gate: bool = False):
        super(CNNNet_PhyChemDi, self).__init__()

        self.dropout_prob = dropout_prob
        self.use_property_gate = use_property_gate

        if self.use_property_gate:
            self.property_gate = nn.Parameter(torch.ones(NUM_PHYSICOCHEMICAL_PROPERTIES))

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

        if self.use_property_gate:
            # [B, 12, 500] * [1, 12, 1] -> [B, 12, 500], broadcast per property
            inputs = inputs * self.property_gate.view(1, -1, 1)

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
