
import torch
import torch_scatter
import torch_geometric as pyg
from . import MLP


class InteractionNetwork(pyg.nn.MessagePassing):
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)
    
    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x = (x, x), edge_feature = edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim = -1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out
    
    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((x_i, x_j, edge_feature), dim= -1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim = self.node_dim, dim_size=dim_size, reduce='sum')
        return (inputs, out)