import json
import argparse
import pickle
import warnings

from utils import load_data
from es_pca.neural_network import NeuralNetwork
from es_pca.neural_network import Solution
from utils import get_split_indices, tranform_data_onehot, create_layers

warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", required=True, type=str)
    args = parser.parse_args()

    with open(args.filename) as json_data:
        data = json.load(json_data)  # data is a python dictionary

    x = load_data(data["dataset"])

    # transform categorical (object type in pandas) columns to one-hot encoded.
    x, num_features_per_network = tranform_data_onehot(x)
    train_indices, val_indices = get_split_indices(x)
    train_x, val_x = x[train_indices], x[val_indices]

    # Instantiate Solution object
    list_layers = [create_layers(n_features, data["n_hidden_layers"], data["hidden_layer_size"], data["activation"]) for n_features in num_features_per_network]
    list_neural_networks = [NeuralNetwork(l) for l in list_layers]
    solution = Solution(list_neural_networks)

    print("Training Baseline")
    obj_list, x_transformed = solution.fit(train_x, val_x, data["sigma"], data["learning_rate"], data["pop_size"], data["alpha_reg_pca"],
                                           data["partial_contribution_objective"], data["num_components"], data["epochs"], data["batch_size"],
                                           data["early_stopping_epochs"])

    pickle.dump((obj_list, x_transformed), open("results/transformed.p", "wb"))


if __name__ == "__main__":

    main()
