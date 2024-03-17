import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.io import arff
import os
import yaml

from typing import Tuple

from es_pca.layers.layers import ForwardLayer, BatchNormLayer
from es_pca.synthetic_datasets import make_two_spheres, make_alternate_stripes, circles_data


def read_arff(path):
    data, meta = arff.loadarff(path)
    data = pd.DataFrame(data)
    return data


def create_scatter_plot(data_transformed, data_pca_transformed):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # Create a 2x1 grid of subplots

    for i, data in enumerate([data_transformed, data_pca_transformed]):
        ax = axes[i]
        ax.grid(True)
        ax.scatter(data[:, 0], data[:, 1], c="blue", s=20, edgecolor="k")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel(r"$\widetilde{z}_1$", size=15)
        ax.set_ylabel(r"$\widetilde{z}_2$", size=15)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()


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


def get_split_indices(data: np.array, random_seed: int, val_prop: float = 0.2) -> Tuple[np.array, np.array]:
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


def transform_data_onehot(data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    object_indices = np.where(data.dtypes == 'object')[0]  # TODO: this means that we curate the data first

    if len(object_indices) == 0:
        return data, [1] * data.shape[1]

    else:

        data_to_one_hot = data.iloc[:, object_indices]

        num_cols_per_categories = list(data_to_one_hot.nunique())
        cols_to_remove = data_to_one_hot.columns

        data_to_one_hot = pd.get_dummies(data_to_one_hot)

        data = data.drop(columns=cols_to_remove, inplace=False)
        num_cols_per_categories = [1] * data.shape[1] + num_cols_per_categories
        data = pd.concat((data_to_one_hot, data), axis=1)

        return data, num_cols_per_categories


def create_nn_for_numerical_col(n_features, n_layers, hidden_size, activation="leaky_relu"):

    layers_list = list()

    layers_list.append(ForwardLayer(n_features, hidden_size, activation))
    layers_list.append(BatchNormLayer(hidden_size))

    for _ in range(n_layers):
        layers_list.append(ForwardLayer(hidden_size, hidden_size, activation))
        layers_list.append(BatchNormLayer(hidden_size))

    layers_list.append(ForwardLayer(hidden_size, 1, 'identity'))

    return layers_list


def create_nn_for_categorical_col(n_features):

    layer = [ForwardLayer(n_features, 1, "identity")]  # simply a single linear layer

    return layer


def create_network(n_features, n_layers, hidden_size, activation="leaky_relu"):

    if n_features > 1:  # for dealing with categorical variables
        layers_list = create_nn_for_categorical_col(n_features)
    else:
        layers_list = create_nn_for_numerical_col(n_features, n_layers, hidden_size, activation=activation)

    return layers_list


def config_load() -> dict:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_dir, "../config_es.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)



