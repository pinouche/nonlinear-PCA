import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.neighbors import LocalOutlierFactor
from scipy.io import arff
import os
import yaml

from typing import Tuple

from es_pca.layers.layers import ForwardLayer, BatchNormLayer
from es_pca.synthetic_datasets import make_two_spheres, make_alternate_stripes, circles_data
from es_pca.data_models.data_models import ConfigDataset


def dataset_config_load(file_path: str, args: argparse.Namespace) -> ConfigDataset:
    """load config file into pydantic object"""
    with open(file_path) as file:
        config_data = yaml.safe_load(file)
        config_data = config_data[args.dataset]

    return ConfigDataset(**config_data)


def remove_files_from_dir(path: str) -> None:
    # Remove all files in the directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def read_arff(path):
    data, meta = arff.loadarff(path)
    data = pd.DataFrame(data)
    return data


def preprocess_data(data: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, np.array]:
    type_class = np.zeros(shape=(1, data.shape[0]))
    if dataset == "abalone":
        type_class = data["sex"]

    elif dataset in ["phoneme", "breast_cancer"]:
        type_class = data["Class"]

    elif dataset in ["wine", "ionosphere", "german_credit", "dermatology", "heart-statlog"]:
        type_class = data["class"]

    return data, type_class


def load_data(dataset: str) -> pd.DataFrame:
    if dataset == "spheres":
        data = make_two_spheres()

    elif dataset == "circles":
        data = circles_data()

    elif dataset == "alternate_stripes":
        data = make_alternate_stripes()

    else:
        path = f"datasets/{dataset}.arff"
        data = read_arff(path)

    return pd.DataFrame(data)


def create_scatter_plot(data_transformed: tuple[np.array, np.array],
                        data_pca_transformed: tuple[np.array, np.array],
                        classes: tuple[np.array, np.array]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # Create a 2x1 grid of subplots

    for i, data in enumerate([data_transformed, data_pca_transformed]):
        ax = axes[i]
        ax.grid(True)
        ax.scatter(data[0][:, 0], data[0][:, 1], c=classes[0], s=20, edgecolor="k", alpha=0.5, label="Training data")
        ax.scatter(data[1][:, 0], data[1][:, 1], c=classes[1], s=20, edgecolor="k", label="Validation data")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', which='major', labelsize=14)

        x_label = r"$\widetilde{x}_1$"
        y_label = r"$\widetilde{x}_2$"
        if i == 1:
            x_label = r"$\widetilde{z}_1$"
            y_label = r"$\widetilde{z}_2$"

        ax.set_xlabel(x_label, size=15)
        ax.set_ylabel(y_label, size=15)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()


def get_split_indices(data: np.array, random_seed: int) -> Tuple[np.array, np.array]:
    config = config_load()
    val_prop = config["val_prop"]
    np.random.seed(random_seed)
    n = data.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[int(n * val_prop):]
    val_indices = indices[:int(n * val_prop)]

    return train_indices, val_indices


def convert_dic_to_list(dictionary: dict) -> list:
    result = []

    for key, value in dictionary.items():
        result.append(value)

    return result


def transform_data_onehot(data: pd.DataFrame, object_indices: list[int]) -> Tuple[pd.DataFrame, list]:
    object_indices = np.array(object_indices)
    data_to_one_hot = data.iloc[:, object_indices]

    num_cols_per_categories = list(data_to_one_hot.nunique())
    cols_to_remove = data_to_one_hot.columns

    data_to_one_hot = pd.get_dummies(data_to_one_hot, columns=cols_to_remove, dtype=int)

    data = data.drop(columns=cols_to_remove, inplace=False)
    num_cols_per_categories = num_cols_per_categories + [1] * data.shape[1]
    data = pd.concat((data_to_one_hot, data), axis=1)

    return data, num_cols_per_categories


def create_nn_for_numerical_col(n_features, n_layers, hidden_size, activation="leaky_relu"):
    layers_list = list()

    layers_list.append(ForwardLayer(n_features, hidden_size, activation))
    # layers_list.append(BatchNormLayer(hidden_size))

    for _ in range(n_layers - 1):
        layers_list.append(ForwardLayer(hidden_size, hidden_size, activation))
        # layers_list.append(BatchNormLayer(hidden_size))

    layers_list.append(ForwardLayer(hidden_size, 1, 'identity'))

    return layers_list


def create_network(n_features, n_layers, hidden_size, activation="leaky_relu"):

    layers_list = create_nn_for_numerical_col(n_features, n_layers, hidden_size, activation=activation)

    return layers_list


def config_load() -> dict:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_dir, "../config_es.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_arguments():
    config = config_load()

    parser = argparse.ArgumentParser(description="test", conflict_handler="resolve")
    parser.add_argument(
        "--dataset",
        default=config["dataset"],
        help="Specify the dataset.",
    )

    parser.add_argument(
        "--partial_contrib",
        type=str,
        default=config["partial_contribution_objective"],
        help="Specify whether or not to use partial contribution objective."
    )

    parser.add_argument(
        "--activation",
        type=str,
        default=config["activation"],
        help="Specify the activation function to use."
    )

    args = parser.parse_args()

    return args


def remove_outliers(data: np.array):
    clf = LocalOutlierFactor(n_neighbors=20)
    mask = clf.fit_predict(data)
    mask[mask == -1] = 0
    data = data[mask.astype(bool)]
    return data
