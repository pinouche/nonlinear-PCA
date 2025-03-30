import pickle
import warnings
import os
from sklearn.preprocessing import StandardScaler
from glob import glob
import re

from loguru import logger
import argparse
import multiprocessing

import pandas as pd
import numpy as np

from utils import load_data, remove_files_from_dir
from es_pca.neural_network.neural_network import NeuralNetwork
from es_pca.neural_network.evolution_strategies import Solution
from es_pca.utils import (get_split_indices, transform_data_onehot, parse_arguments, config_load,
                          dataset_config_load, preprocess_data)
from es_pca.data_models.data_models import ConfigDataset
from es_pca.layers.init_weights_layers import create_nn_for_numerical_col

warnings.filterwarnings("ignore")


def main(config_es: dict, dataset_config: ConfigDataset, args: argparse.Namespace, run_index: int) -> None:
    if args.partial_contrib == "false":
        args.partial_contrib = False
    elif args.partial_contrib == "true":
        args.partial_contrib = True
    else:
        raise ValueError(f"{args.partial_contrib} not a boolean.")

    x = load_data(args.dataset)
    x = x.dropna()

    classes = np.zeros(x.shape[0])

    if args.dataset not in ["circles", "spheres", "alternate_stripes"]:
        x, classes = preprocess_data(x, args.dataset)
        classes, mapping = pd.factorize(classes)

    logger.info(f"The column types of the dataset are: {x.dtypes}")

    # transform categorical columns to one-hot encoded.
    if dataset_config.categorical_features:
        x, num_features_per_network = transform_data_onehot(x,
                                                            dataset_config.categorical_features
                                                            )
    else:
        num_features_per_network = np.array([1] * x.shape[1])

    train_indices, val_indices = get_split_indices(x, run_index)
    train_x = x.iloc[train_indices].values
    val_x = x.iloc[val_indices].values

    if args.dataset not in ["circles", "spheres", "alternate_stripes"]:
        scaler = StandardScaler()
        scaler.fit(x.iloc[train_indices])
        train_x = scaler.transform(train_x)
        val_x = scaler.transform(val_x)

    y = classes[train_indices], classes[val_indices]

    # Instantiate Solution object or load pre-existing list[NeuralNetwork]

    dataset_folder = "real_world_data"
    if args.dataset in ["circles", "spheres", "alternate_stripes"]:
        dataset_folder = "synthetic_data"

    # Construct the directory path (excluding the epoch part)
    directory_path = (f"results/datasets/{dataset_folder}/{args.dataset}/"
                      f"activation={args.activation}/"
                      f"partial_contrib={str(args.partial_contrib)}/{str(run_index)}/")

    # Search for files matching the pattern
    file_pattern = os.path.join(directory_path, "best_individual_epoch_*.p")
    files = glob(file_pattern)

    latest_epoch = 0
    previous_results = []
    if files:
        file = None
        for doc in files:
            match = re.search(r"best_individual_epoch_(\d+)\.p", doc)
            if match:
                file = doc
                latest_epoch = int(match.group(1))
                break  # Since we know there's only one match, we can exit the loop

        if latest_epoch > config_es["epochs"]:
            raise ValueError(f"Latest saved epoch {latest_epoch} is greater than the number of epochs {config_es['epochs']}")

        if file:
            print(f"Latest saved epoch found: {latest_epoch}")
            with open(file, "rb") as f:
                list_neural_networks = pickle.load(f)
            # Delete the file after loading
            os.remove(file)
            print(f"File {file} has been deleted")

            # Load previous results if they exist
            results_file_path = os.path.join(directory_path, "results_list.p")

            if os.path.exists(results_file_path):
                with open(results_file_path, "rb") as f:
                    previous_results = pickle.load(f)
                print(f"Loaded previous results up to epoch {latest_epoch}, length {len(previous_results)}")

    else:
        list_neural_networks = [NeuralNetwork(create_nn_for_numerical_col(n_features,
                                                                          config_es["n_hidden_layers"],
                                                                          config_es["hidden_layer_size"],
                                                                          args.activation,
                                                                          config_es["init_mode"])) for n_features in
                                num_features_per_network]

    solution = Solution(list_neural_networks)

    logger.info(f"Run number {run_index} training baseline for dataset={args.dataset}, "
                f"partial_contrib={args.partial_contrib}, "
                f"activation_function={args.activation}")

    results = solution.fit(train_x,
                           val_x,
                           y,
                           config_es["sigma"],
                           config_es["learning_rate"],
                           config_es["pop_size"],
                           args.partial_contrib,
                           config_es["num_components"],
                           config_es["epochs"],
                           config_es["batch_size"],
                           config_es["early_stopping_epochs"],
                           train_indices,
                           val_indices,
                           run_index,
                           latest_epoch,
                           config_es["plot"]
                           )

    if previous_results:
        results_list = previous_results + results
    else:
        results_list = results

    saving_path = (f"results/datasets/{dataset_folder}/{args.dataset}/"
                   f"activation={args.activation}/"
                   f"partial_contrib={str(args.partial_contrib)}/{str(run_index)}/results_list.p")

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
    tmp_path = "./tmp_files/"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    remove_files_from_dir(tmp_path)

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
