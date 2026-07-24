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


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    property_file = (
        project_root
        / "data"
        / "Physicochemical_properties_Di.xlsx"
    )

    physicochemical_table = (
        load_physicochemical_properties_di(
            property_file
        )
    )

    test_sequence = (
        "A" * 250
        + "CG"
        + "T" * 249
    )

    test_features_numpy = (
        convertSampleToPhyChemVector_Di(
            sampleSeq=[test_sequence],
            PhyChemPropTable_Di=physicochemical_table,
            expected_length=501,
        )
    )

    test_features_tensor = torch.from_numpy(
        test_features_numpy
    ).float()

    model = CNNNet_PhyChemDi(
        dropout_prob=0.0
    )

    model.eval()

    with torch.no_grad():
        conv1_output = model.conv1(
            test_features_tensor
        )
        pool1_output = model.pool1(
            conv1_output
        )

        conv2_output = model.conv2(
            pool1_output
        )
        pool2_output = model.pool2(
            conv2_output
        )

        conv3_output = model.conv3(
            pool2_output
        )
        pool3_output = model.pool3(
            conv3_output
        )

        final_output = model(
            test_features_tensor
        )

    print(
        "Input shape:",
        tuple(test_features_tensor.shape),
    )

    print(
        "After Conv1:",
        tuple(conv1_output.shape),
    )

    print(
        "After Pool1:",
        tuple(pool1_output.shape),
    )

    print(
        "After Conv2:",
        tuple(conv2_output.shape),
    )

    print(
        "After Pool2:",
        tuple(pool2_output.shape),
    )

    print(
        "After Conv3:",
        tuple(conv3_output.shape),
    )

    print(
        "After Pool3:",
        tuple(pool3_output.shape),
    )

    print(
        "Flattened output shape:",
        tuple(final_output.shape),
    )

    assert test_features_tensor.shape == (
        1,
        12,
        500,
    )

    assert pool3_output.shape == (
        1,
        8,
        60,
    )

    assert final_output.shape == (
        1,
        480,
    )

    print(
        "\nPhysicochemical CNN shape test passed."
    )