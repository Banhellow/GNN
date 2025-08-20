import torch
import math

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, layer_norm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        self.layers.append(torch.nn.ReLU())
        for layer in range(layers - 2):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            if layer != layers - 3:
                self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_size, output_size))
        if layer_norm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)
    
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data 