import torch
import numpy as np
from torch import nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, n_layers, layer_size_list, input_size, output_size, n_components_to_keep):
        super(NeuralNetwork, self).__init__()

        self.layer_size_list = layer_size_list
        self.n_layers = n_layers
        self.output_size = output_size
        self.input_size = input_size
        self.k = n_components_to_keep

        assert len(self.layer_size_list) == n_layers, f"`layer_size_list` should have length {n_layers}"

        # hidden layers
        self.shared_hidden_layers_list = nn.ModuleList()
        for l in range(self.n_shared_hidden_layers):
            self.shared_hidden_layers_list.append(nn.LazyLinear(self.shared_hidden_size[l]))
            self.shared_hidden_layers_list.append(nn.SELU())
            if dropout_montecarlo:
                self.shared_hidden_layers_list.append(DropoutAlwaysActivated(self.drop))
            else:
                self.shared_hidden_layers_list.append(nn.Dropout(self.drop))
            self.shared_hidden_layers_list.append(nn.BatchNorm1d(self.shared_hidden_size[l]))

        self.shared_hidden_layers_list = nn.Sequential(*self.shared_hidden_layers_list)

        if n_non_shared_hidden_layers > 0:
            self.non_shared_concat = nn.ModuleList(nn.ModuleList() for _ in range(self.output_size))
            for i in range(self.output_size):
                for l in range(self.n_non_shared_hidden_layers):
                    self.non_shared_concat[i].append(nn.LazyLinear(self.non_shared_hidden_size[l]))
                    self.non_shared_concat[i].append(nn.SELU())
                    self.non_shared_concat[i].append(nn.Dropout(self.drop))
                    self.non_shared_concat[i].append(nn.BatchNorm1d(self.non_shared_hidden_size[l]))

                self.non_shared_concat[i].append(nn.LazyLinear(1))
                self.non_shared_concat[i] = nn.Sequential(*self.non_shared_concat[i])
        else:
            self.output_layer = nn.LazyLinear(self.output_size)

    def forward(self, x):

        # x = self.custom_layer_input_dropout(x)

        if indices_to_split is not None:
            x_non_shared = np.split(x[:, self.shared_input_size:], self.indices_to_split, 1)
            x = x[:, :self.shared_input_size]

        shared_x = self.shared_hidden_layers_list(x)

        if n_non_shared_hidden_layers > 0:
            output = torch.empty(size=(shared_x.shape[0], self.output_size)).to("cuda")
            for i in range(self.output_size):
                if indices_to_split is not None:
                    not_shared_x = torch.hstack((shared_x, x_non_shared[i]))
                else:
                    not_shared_x = shared_x
                not_shared_x = self.non_shared_concat[i](not_shared_x)

                output[:, i] = torch.squeeze(not_shared_x)

        else:
            output = self.output_layer(shared_x)

        return output
