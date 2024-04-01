import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn


class GraphPredictionModel(nn.Module):
    """A graph prediction network with average pooling before"""
    def __init__(self,
                 c_out: int,
                 gnn_layers: nn.Module,
                 dp_rate: float) -> None:
        """Initialise the object.

        :params c_out_gnn: Number of output channel from the GNN layers
        :params c_out: The graph's output channel
        :params gnn_layers: a Graph Net layer object (see: src/models/components folder)
        :params dp_rate: Drop out rate for the linear layer
        """
        super().__init__()

        self.gnn = gnn_layers
        self.head = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(self.gnn.layers[-1].out_channels, c_out)
        )
        self.c_out = c_out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Single forward pass

        :params x: Input features per node
        :params edge_index: List of vertex index pairs representing the edges in the graph
        :params batch_idx: Index of batch element for each node
        """
        x = self.gnn(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)

        return x
