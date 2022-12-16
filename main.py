import numpy as np
import warnings
warnings.filterwarnings("ignore")

from datasets.synthetic_datasets import load_data
from layers.layers import ForwardLayer, BatchNormLayer
from neural_network.neural_network import NeuralNetwork
from neural_network.evolution_strategies import Solution


def main():

    # neural network params
    input_size = 1
    output_size = 1
    hidden_layer_size = 16
    activation = 'selu'

    # ES params
    pop_size = 1000
    sigma = 0.001
    learning_rate = 0.001
    epochs = 100
    batch_size = 128

    # objective function params
    partial_contribution_objective = False
    num_components = 1

    dataset = 'circles'
    x = load_data(dataset)

    # neural network
    list_layers = [[ForwardLayer(input_size, hidden_layer_size, activation),
                    BatchNormLayer(hidden_layer_size),
                    ForwardLayer(hidden_layer_size, hidden_layer_size, activation),
                    ForwardLayer(hidden_layer_size, hidden_layer_size, activation),
                    ForwardLayer(hidden_layer_size, output_size, 'identity')] for _ in range(x.shape[1])]

    list_neural_networks = [NeuralNetwork(l) for l in list_layers]
    solution = Solution(list_neural_networks)

    print("Training Baseline")
    obj_list, x_transformed = solution.fit(x, sigma, learning_rate, pop_size, partial_contribution_objective, num_components, epochs, batch_size)
    print(obj_list)


if __name__ == "__main__":

    main()
