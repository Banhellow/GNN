import torch
import torch_geometric as pyg
from . import MLP
from . import InteractionNetwork

class LearningSimulator(torch.nn.Module):
    def __init__(self, hidden_size, gnn_layers, particle_types_dim, dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.embed_type = torch.nn.Embedding(5, particle_types_dim)
        self.node_in = MLP(particle_types_dim + dim * window_size, hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim + 1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layer_norm=False)
        self.gnn_layers = gnn_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(hidden_size, 3) for _ in range(gnn_layers)])
        self.reset_params()
        
    def reset_params(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        embed_tensor =self.embed_type(data.x)
        node_feature = torch.cat((embed_tensor, data.pos), dim = -1).to(torch.float32)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr.to(torch.float32))
        for i in range(self.gnn_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)

        out = self.node_out(node_feature)
        return out