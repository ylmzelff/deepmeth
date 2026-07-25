import math

import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Graph convolution layer used in ncVarPred.

    Input:
        node_features: [number_of_nodes, in_features]
        adjacency:     [number_of_nodes, number_of_nodes]

    Output:
        [number_of_nodes, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.FloatTensor(
                in_features,
                out_features,
            )
        )

        if bias:
            self.bias = Parameter(
                torch.FloatTensor(out_features)
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

        self.reset_parameters()

    def reset_parameters(self):
        # Same initialization used by the original ncVarPred code.
        stdv = 1.0 / math.sqrt(
            self.weight.size(1)
        )

        self.weight.data.uniform_(
            -stdv,
            stdv,
        )

        if self.bias is not None:
            self.bias.data.uniform_(
                -stdv,
                stdv,
            )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features:
                Node feature matrix [N, F_in].

            adjacency:
                Normalized adjacency matrix [N, N].

        Returns:
            Updated node representations [N, F_out].
        """

        # [N, F_in] x [F_in, F_out]
        # -> [N, F_out]
        support = torch.mm(
            node_features,
            self.weight,
        )

        # [N, N] x [N, F_out]
        # -> [N, F_out]
        output = torch.spmm(
            adjacency,
            support,
        )

        if self.bias is not None:
            return output + self.bias

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
