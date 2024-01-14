import yaml
import pickle
import warnings

import numpy as np

from utils import load_data
from es_pca.neural_network.neural_network import NeuralNetwork
from es_pca.neural_network.evolution_strategies import Solution
from es_pca.utils import get_split_indices, transform_data_onehot, create_network

warnings.filterwarnings("ignore")


def main(config_es: dict, dataset_config: dict) -> None:

    x = load_data(config_es["dataset"])
    print(type(x), x.shape)

    # transform categorical (object type in pandas) columns to one-hot encoded.
    x, num_features_per_network = transform_data_onehot(x)
    train_indices, val_indices = get_split_indices(x)
    train_x, val_x = np.array(x.iloc[train_indices]), np.array(x.iloc[val_indices])

    # Instantiate Solution object
    list_neural_networks = [NeuralNetwork(create_network(n_features,
                                                         config_es["n_hidden_layers"],
                                                         config_es["hidden_layer_size"],
                                                         config_es["activation"])) for n_features in
                            num_features_per_network]
    solution = Solution(list_neural_networks)

    print("Training Baseline...")
    obj_list, x_transformed = solution.fit(train_x, val_x, config_es["sigma"], config_es["learning_rate"],
                                           config_es["pop_size"], config_es["alpha_reg_pca"],
                                           config_es["partial_contribution_objective"], config_es["num_components"],
                                           config_es["epochs"], config_es["batch_size"],
                                           config_es["early_stopping_epochs"])

    pickle.dump((obj_list, x_transformed), open("results/transformed.p", "wb"))


if __name__ == "__main__":

    with open("./config_es.yaml", "r") as config_data:
        config_evo = yaml.safe_load(config_data)

    with open("./datasets_config.yaml", "r") as config_data:
        config_data = yaml.safe_load(config_data)
        config_data = config_data[config_evo["dataset"]]

    main(config_evo, config_data)
