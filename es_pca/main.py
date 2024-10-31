import pickle
import warnings
import os
from loguru import logger
import argparse

import numpy as np

from utils import load_data
from es_pca.neural_network.neural_network import NeuralNetwork
from es_pca.neural_network.evolution_strategies import Solution
from es_pca.utils import (get_split_indices, transform_data_onehot, create_network, parse_arguments, config_load,
                          dataset_config_load)
from es_pca.data_models.data_models import ConfigDataset

warnings.filterwarnings("ignore")


def main(config_es: dict, dataset_config: ConfigDataset, args: argparse.Namespace, run_index: int) -> None:

    if args.partial_contrib == "false":
        args.partial_contrib = False
    elif args.partial_contrib == "true":
        args.partial_contrib = True
    else:
        raise ValueError(f"Partial contrib should be in ['false', 'true'], got {args.partial_contrib}.")

    x = load_data(args.dataset)

    logger.info(f"The column types of the dataset are: {x.dtypes}")

    # transform categorical (object type in pandas) columns to one-hot encoded.
    x, num_features_per_network = transform_data_onehot(x)
    train_indices, val_indices = get_split_indices(x, run_index)
    train_x, val_x = np.array(x.iloc[train_indices]), np.array(x.iloc[val_indices])

    # Instantiate Solution object
    list_neural_networks = [NeuralNetwork(create_network(n_features,
                                                         config_es["n_hidden_layers"],
                                                         config_es["hidden_layer_size"],
                                                         args.activation)) for n_features in
                            num_features_per_network]
    solution = Solution(list_neural_networks)

    logger.info(f"Run number {run_index} training baseline for dataset={args.dataset}, "
                f"partial_contrib={args.partial_contrib}, "
                f"activation_function={args.activation}")

    obj_list, x_transformed = solution.fit(train_x, val_x,
                                           config_es["sigma"],
                                           config_es["learning_rate"],
                                           config_es["pop_size"],
                                           args.partial_contrib,
                                           config_es["num_components"],
                                           config_es["epochs"],
                                           config_es["batch_size"],
                                           config_es["early_stopping_epochs"],
                                           config_es["plot"])

    saving_path = f"results/{args.dataset}/activation={args.activation}/partial_contrib={str(args.partial_contrib)}/{str(run_index)}.p"

    if not os.path.exists(os.path.dirname(saving_path)):
        os.makedirs(os.path.dirname(saving_path))

    pickle.dump((obj_list, x_transformed), open(saving_path, "wb"))


if __name__ == "__main__":

    # this is the config for the evolution strategies run
    config_evo = config_load()
    args = parse_arguments()

    # this is the config for the datasets specifications
    config_data = dataset_config_load("./datasets_config.yaml", args)

    number_of_runs = config_evo["number_of_runs"]

    for i in range(number_of_runs):
        main(config_evo, config_data, args, i)
