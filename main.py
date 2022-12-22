import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from datasets.load_data import load_data
from neural_network.neural_network import NeuralNetwork
from neural_network.evolution_strategies import Solution

from utils import split_data, tranform_data_onehot, create_layers


def main():

    # neural network params
    hidden_layer_size = 32
    activation = 'leaky_relu'
    n_hidden_layers = 1

    # ES params
    pop_size = 500
    sigma = 0.01
    learning_rate = 0.0001
    epochs = 100
    batch_size = 128
    early_stopping_epochs = 20

    # objective function params
    alpha_reg_pca = 0
    partial_contribution_objective = True
    num_components = 1

    dataset = 'spheres'
    x = load_data(dataset)
    x, num_features_per_network = tranform_data_onehot(x)  # transform categorical (object type in pandas) columns to one-hot encoded.
    train_x, val_x = split_data(x)

    # Instantiate Solution object
    list_layers = [create_layers(n_features, n_hidden_layers, hidden_layer_size, activation) for n_features in num_features_per_network]
    list_neural_networks = [NeuralNetwork(l) for l in list_layers]
    solution = Solution(list_neural_networks)

    print("Training Baseline")
    obj_list, x_transformed = solution.fit(train_x, val_x, sigma, learning_rate, pop_size, alpha_reg_pca, partial_contribution_objective, num_components,
                                           epochs, batch_size, early_stopping_epochs)

    pickle.dump(x_transformed, open("../../Documents/x_transformed.p", "wb"))


if __name__ == "__main__":

    main()
