import numpy as np
import pickle
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
    hidden_layer_size = 64
    bottleneck_layer_size = 16
    activation = 'relu'

    # ES params
    pop_size = 1000
    sigma = 0.01
    learning_rate = 0.001
    epochs = 10
    batch_size = 64
    train = True

    # objective function params
    alpha_reg_pca = 0.01
    partial_contribution_objective = False
    num_components = 1

    dataset = 'circles'
    x = load_data(dataset)

    # neural network
    list_layers = [[ForwardLayer(input_size, hidden_layer_size, activation),
                    BatchNormLayer(hidden_layer_size),
                    ForwardLayer(hidden_layer_size, bottleneck_layer_size, activation),
                    BatchNormLayer(bottleneck_layer_size),
                    ForwardLayer(bottleneck_layer_size, hidden_layer_size, activation),
                    BatchNormLayer(hidden_layer_size),
                    ForwardLayer(hidden_layer_size, output_size, 'identity')] for _ in range(x.shape[1])]

    list_neural_networks = [NeuralNetwork(l) for l in list_layers]
    solution = Solution(list_neural_networks)

    print("Training Baseline")
    obj_list, x_transformed = solution.fit(x, sigma, learning_rate, pop_size, alpha_reg_pca, partial_contribution_objective, num_components,
                                           epochs, batch_size, train)

    pickle.dump(x_transformed, open("../../Documents/x_transformed.p", "wb"))


if __name__ == "__main__":

    main()
