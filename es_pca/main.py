import pickle
import warnings
import os
from loguru import logger
import argparse
import multiprocessing
import shutil

import pandas as pd
import numpy as np

from utils import load_data
from es_pca.neural_network.neural_network import NeuralNetwork
from es_pca.neural_network.evolution_strategies import Solution
from es_pca.utils import (get_split_indices, transform_data_onehot, create_network, parse_arguments, config_load,
                          dataset_config_load, preprocess_data)
from es_pca.data_models.data_models import ConfigDataset

warnings.filterwarnings("ignore")


def main(config_es: dict, dataset_config: ConfigDataset, args: argparse.Namespace, run_index: int) -> None:
    if args.partial_contrib == "false":
        partial_contrib = False
    elif args.partial_contrib == "true":
        partial_contrib = True
    else:
        raise ValueError(f"Partial contrib should be in ['false', 'true'], got {args.partial_contrib}.")

    x = load_data(args.dataset)
    x = x.dropna()
    x, classes = preprocess_data(x, args.dataset)
    classes, mapping = pd.factorize(classes)

    logger.info(f"The column types of the dataset are: {x.dtypes}")

    # transform categorical (object type in pandas) columns to one-hot encoded.
    if dataset_config.categorical_features:
        x, num_features_per_network = transform_data_onehot(x, dataset_config.categorical_features)
    else:
        num_features_per_network = np.array([1] * x.shape[1])

    # split train and validation
    train_indices, val_indices = get_split_indices(x, run_index)

    print(train_indices.shape, val_indices.shape)

    train_x, val_x = np.array(x.iloc[train_indices]), np.array(x.iloc[val_indices])
    y = classes[train_indices], classes[val_indices]

    # Instantiate Solution object
    list_neural_networks = [NeuralNetwork(create_network(n_features,
                                                         config_es["n_hidden_layers"],
                                                         config_es["hidden_layer_size"],
                                                         args.activation)) for n_features in
                            num_features_per_network]
    solution = Solution(list_neural_networks)

    logger.info(f"Run number {run_index} training baseline for dataset={args.dataset}, "
                f"partial_contrib={partial_contrib}, "
                f"activation_function={args.activation}")

    results_list = solution.fit(train_x,
                                val_x,
                                y,
                                config_es["sigma"],
                                config_es["learning_rate"],
                                config_es["pop_size"],
                                partial_contrib,
                                config_es["num_components"],
                                config_es["epochs"],
                                config_es["batch_size"],
                                config_es["early_stopping_epochs"],
                                train_indices,
                                val_indices,
                                run_index,
                                config_es["plot"]
                                )

    saving_path = f"results/{args.dataset}/activation={args.activation}/partial_contrib={str(partial_contrib)}/{str(run_index)}.p"

    if not os.path.exists(os.path.dirname(saving_path)):
        os.makedirs(os.path.dirname(saving_path))

    pickle.dump(results_list, open(saving_path, "wb"))


def run_single_iteration(args: argparse.Namespace):
    """
    Wrapper function to unpack arguments for multiprocessing
    """
    evo_config, data_config, arguments, index = args
    main(evo_config, data_config, arguments, index)


if __name__ == "__main__":

    # make sure this temp dir is empty (with no artifacts from the run on the previous dataset)
    if not os.path.exists("tmp_files/"):
        os.makedirs(os.path.dirname("tmp_files/"))
    shutil.rmtree("tmp_files/")

    # Load configurations
    config_evo = config_load()
    args = parse_arguments()
    config_data = dataset_config_load("./datasets_config.yaml", args)

    # Get number of runs
    number_of_runs = config_evo["number_of_runs"]

    # Determine the number of processes (you can adjust this)
    num_processes = min(multiprocessing.cpu_count(), number_of_runs)

    # Prepare arguments for each run
    run_args = [
        (config_evo, config_data, args, i) for i in range(number_of_runs)
    ]

    # Use multiprocessing Pool to run iterations in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_single_iteration, run_args)
