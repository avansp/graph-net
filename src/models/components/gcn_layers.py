from torch import nn
import torch_geometric.nn as geom_nn


class GCNLayers(nn.Module):
    """Layers of GCN models.

    Note: self.layers is nn.ModuleList object.
    """
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        num_layers: int = 2,
        dp_rate: float = 0.1
    ) -> None:
        """Initialise graph network model.

        :param c_in: Dimension of the input features.
        :param c_hidden: Dimension of the hidden features.
        :param c_out: Dimension of the output features.
        :param num_layers: Number of hidden graph layers.
        :param dp_rate: Dropout rate to apply throughout the network.
        """
        super().__init__()

        # create intermediate layers from 0 ... (num_layers-1)
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers-1):
            layers += [
                geom_nn.GCNConv(in_channels=in_channels, out_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        # add the last layer
        layers += [geom_nn.GCNConv(in_channels=in_channels, out_channels=c_out)]

        # final module
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """Perform a single forward pass through the network.

        :param x: Input features per node
        :param edge_index: List of vertex index pairs representing the edges in the graph
        """
        for l in self.layers:
            # For graph layers, we need to add "edge_index" tensor as additional input.
            # All PYG graph layer inherits the class "MessagePassing", which we can use it to check
            # if a layer is a graph layer.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)

        return x
