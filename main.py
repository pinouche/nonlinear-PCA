import numpy as np

from datasets.synthetic_datasets import load_data
from layers.layers import ForwardLayer, BatchNormLayer
from neural_network.neural_network import NeuralNetwork
from neural_network.evolution_strategies import Solution


def main():

    input_size = 1
    output_size = 1
    hidden_layer_size = 32
    activation = 'selu'

    pop_size = 1000

    dataset = 'circles'
    x = load_data(dataset)

    # neural network
    list_layers = [[ForwardLayer(input_size, hidden_layer_size, activation),
                   ForwardLayer(hidden_layer_size, hidden_layer_size, activation),
                   ForwardLayer(hidden_layer_size, output_size, 'identity')] for _ in range(x.shape[1])]

    list_neural_networks = [NeuralNetwork(l) for l in list_layers]
    solution = Solution(list_neural_networks)

    pass


if __name__ == "__main__":

    main()
