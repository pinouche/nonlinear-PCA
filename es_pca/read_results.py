import os
import pickle
import re
from typing import List, Any, Dict, Optional
import argparse
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Make sure the script can find the utils module
from es_pca.utils import load_data, preprocess_data, transform_data_onehot, config_load, dataset_config_load, parse_arguments
from es_pca.neural_network.evolution_strategies import Solution

# Load configurations
config_evo = config_load()


def prepare_data(dataset: str, train_idx: np.array, val_idx: np.array):

    args = parse_arguments()
    dataset_config = dataset_config_load("./datasets_config.yaml", args)

    x = load_data(dataset)
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

    train_x = x.iloc[train_idx].values
    val_x = x.iloc[val_idx].values

    if args.dataset not in ["circles", "spheres", "alternate_stripes"]:
        scl = StandardScaler()
        scl.fit(train_x)
        train_x = scl.transform(train_x)
        val_x = scl.transform(val_x)

    y = classes[train_indices], classes[val_indices]

    return train_x, val_x, y, num_features_per_network


def find_latest_individual_file(run_path: str) -> Optional[str]:
    """Finds the 'best_individual' file with the highest epoch number."""
    individual_files = [
        f
        for f in os.listdir(run_path)
        if f.startswith("best_individual") and f.endswith(".p")
    ]

    latest_file = None
    max_epoch = -1

    for f in individual_files:
        match = re.search(r"epoch_(\d+)\.p$", f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = f

    return os.path.join(run_path, latest_file) if latest_file else None


def load_and_pair_results(
    base_folder: str, partial_contrib: str
) -> List[Dict[str, Any]]:
    """
    For each run, loads the last epoch from 'results_list.p' and the single
    latest 'best_individual*.p' file.

    Returns:
        A list of dictionaries, each representing a run with its paired final results.
    """
    all_runs_data = []
    search_path_base = os.path.join(base_folder)

    run_parent_dirs = []
    for root, dirs, _ in os.walk(search_path_base):
        if f"partial_contrib={partial_contrib}" in root and any(
            d.isdigit() for d in dirs
        ):
            run_parent_dirs.append(root)

    if not run_parent_dirs:
        return []

    for run_parent_dir in run_parent_dirs:
        print(f"\nSearching for runs in: {run_parent_dir}")
        for item in sorted(os.listdir(run_parent_dir)):
            run_path = os.path.join(run_parent_dir, item)
            if os.path.isdir(run_path) and item.isdigit():
                run_data = {"run_id": int(item), "path": run_path}

                # 1. Load the latest 'best_individual' file
                latest_individual_file = find_latest_individual_file(run_path)
                if latest_individual_file:
                    try:
                        with open(latest_individual_file, "rb") as f:
                            run_data["best_individual_data"] = pickle.load(f)
                    except (pickle.UnpicklingError, EOFError) as e:
                        print(f"  Could not read pickle file {latest_individual_file}: {e}")

                # 2. Load the last element from 'results_list.p'
                results_file = os.path.join(run_path, "results_list.p")
                if os.path.exists(results_file):
                    try:
                        with open(results_file, "rb") as f:
                            results_list = pickle.load(f)
                            if results_list:
                                run_data["latest_epoch_data"] = results_list[-1]
                    except (pickle.UnpicklingError, EOFError) as e:
                        print(f"  Could not read pickle file {results_file}: {e}")

                # Only add if we have the essential data
                if "latest_epoch_data" in run_data and "best_individual_data" in run_data:
                    all_runs_data.append(run_data)

    return all_runs_data


def display_run_data(run_data: Dict[str, Any]):
    """Helper function to print the final results of a single run."""
    print(f"\n--- Analysis for Run ID: {run_data['run_id']} ---")

    print(f"  - Paired 'best_individual' with the final epoch from 'results_list'.")

    latest_epoch_data = run_data["latest_epoch_data"]
    neural_network_object = run_data["best_individual_data"]
    try:
        model_info, objectives = latest_epoch_data
        pca_model, scaler, train_indices, val_indices = model_info
        print("  - Final Epoch Data:")
        print(f"    - NN Object:        {neural_network_object}")
        print(f"    - PCA Model:         {type(pca_model)}")
        print(f"    - Scaler:            {type(scaler)}")
        print(f"    - Train Indices:     {train_indices.shape}")
        print(f"    - Validation Indices:{val_indices.shape}")
        print(f"    - Final Objectives:  Train={objectives[0]:.4f}, Val={objectives[1]:.4f}")
    except (ValueError, TypeError) as e:
        print(f"    - Could not unpack latest epoch data. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read and process final result files for a given dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset (e.g., 'ionosphere', 'circles').",
    )
    parser.add_argument(
        "--partial_contrib",
        type=str,
        choices=["true", "false"],
        required=True,
        help="Specify whether partial contribution was used ('true' or 'false').",
    )
    args = parser.parse_args()

    synthetic_datasets = ["circles", "spheres", "alternate_stripes"]
    dataset_type_folder = (
        "synthetic_data" if args.dataset in synthetic_datasets else "real_world_data"
    )
    base_folder_path = os.path.join(
        "results", "datasets", dataset_type_folder, args.dataset
    )
    partial_contrib_value = "True" if args.partial_contrib == "true" else "False"

    if not os.path.isdir(base_folder_path):
         raise ValueError(f"The specified dataset folder '{base_folder_path}' does not exist.")

    all_paired_data = load_and_pair_results(base_folder_path, partial_contrib_value)

    if all_paired_data:
        print(f"\nFound and processed final data for {len(all_paired_data)} runs.")
        for run_data_item in all_paired_data:
            display_run_data(run_data_item)
        print("\nAll final results are loaded and available in the 'all_paired_data' variable.")
    else:
        print("\nNo complete run data could be loaded for the specified configuration.")

    for data_dict in all_paired_data:

        epoch_data = data_dict["latest_epoch_data"]
        neural_network = data_dict["best_individual_data"]

        model_info, objectives = epoch_data
        pca_model, scaler, train_indices, val_indices = model_info
        neural_network_solution = Solution(neural_network)

        train_x, val_x, y, num_features_per_network = prepare_data(args.dataset,
                                                                   train_indices,
                                                                   val_indices)

        x_transformed_train = neural_network_solution.predict(train_x)
        x_transformed_val = neural_network_solution.predict(val_x)

        training_data_transformed = scaler.transform(x_transformed_train)
        pca_transformed_data = pca_model.transform(training_data_transformed)

        val_data_transformed = scaler.transform(x_transformed_val)
        pca_transformed_val_data = pca_model.transform(val_data_transformed)

        plt.plot()
        plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], c=y[0], cmap='viridis')
        plt.scatter(pca_transformed_val_data[:, 0], pca_transformed_val_data[:, 1], c=y[1], cmap='viridis')
        plt.show()