import os
import pickle
import re
from typing import List, Any, Dict, Optional
import argparse
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from numpy import ndarray

from es_pca.utils import config_load
from es_pca.neural_network.evolution_strategies import Solution

# Load configurations
config_evo = config_load()


def create_biplot(pca: PCA,
                  data: ndarray,
                  original_column_names: list[str] | None = None):

    # Create the plot with improved aesthetics
    plt.figure(figsize=(10, 8))

    plt.scatter(data[:, 0], data[:, 1],
                alpha=0.75, edgecolor='black', linewidth=1)

    # Variable (feature) arrows
    components = pca.components_[0:2, :]
    for i, feature in enumerate(original_column_names):
        # Straight arrows
        plt.arrow(0, 0, components[0, i], components[1, i],
                  color='red',
                  head_width=0.05,
                  head_length=0.1,
                  alpha=0.7)

        # Add feature labels with slight offset
        plt.text(components[0, i] * 1.2,
                 components[1, i] * 1.2,
                 feature,
                 color='darkred',
                 ha='center',
                 va='center',
                 fontweight='bold',
                 fontsize=12)

    # Styling
    plt.title('PCA Biplot', fontsize=15)
    plt.xlabel(r"$\widetilde{z}_1$" + f" (variance explained: {pca.explained_variance_ratio_[1] * 100:.2f}%)",
               fontsize=12)
    plt.ylabel(r"$\widetilde{z}_2$" + f" (variance explained: {pca.explained_variance_ratio_[1] * 100:.2f}%)",
               fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    # Add discrete legend
    plt.legend(fontsize=12)
    plt.show()


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

                # k=1. Load the latest 'best_individual' file
                latest_individual_file = find_latest_individual_file(run_path)
                if latest_individual_file:
                    try:
                        with open(latest_individual_file, "rb") as f:
                            run_data["best_individual_data"] = pickle.load(f)
                    except (pickle.UnpicklingError, EOFError) as e:
                        print(f"  Could not read pickle file {latest_individual_file}: {e}")

                # k=2. Load the last element from 'results_list.p'
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
        pca_model, scaler, train_indices, val_indices, pca_transformed_val, pca_transformed_train = model_info
        print("  - Final Epoch Data:")
        print(f"    - NN Object:        {neural_network_object}")
        print(f"    - PCA Model:         {type(pca_model)}")
        print(f"    - Scaler:            {type(scaler)}")
        print(f"    - PCA transformed data train: {pca_transformed_train.shape}")
        print(f"    - PCA transformed data val: {pca_transformed_val.shape}")
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

        pca_model, scaler, train_indices, val_indices, pca_transformed_val, pca_transformed_train = model_info
        neural_network_solution = Solution(neural_network)

        original_column_names = labels = [fr"$\Phi_{{{i+1}}}(x_{{{i+1}}})$" for i in range(len(pca_model.explained_variance_ratio_))]

        create_biplot(pca_model, pca_transformed_train, original_column_names)
