import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from es_pca.layers import ForwardLayer, BatchNormLayer

from es_pca.synthetic_datasets import make_two_spheres, make_alternate_stripes, circles_data


def load_data(dataset):

    if dataset == "spheres":
        data = make_two_spheres()

    elif dataset == "circles":
        data = circles_data()

    elif dataset == "alternate_stripes":
        data = make_alternate_stripes()

    elif dataset == "abalone":
        data = pd.read_csv("../datasets/abalone.data")

    else:
        raise ValueError(f"dataset {dataset} is not valid.")

    return data


def get_split_indices(data, val_prop: float = 0.2):

    n = data.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[int(n * val_prop):]
    val_indices = indices[:int(n * val_prop)]

    return train_indices, val_indices


def tranform_data_onehot(data):

    data = pd.DataFrame(data)  # here, we decide to work with data frames, so we convert to df
    object_indices = np.where(data.dtypes == 'object')[0]
    data_to_one_hot = data.iloc[:, object_indices]

    num_cols_per_categories = list(data_to_one_hot.nunique())
    cols_to_remove = data_to_one_hot.columns

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data_to_one_hot)

    data_to_one_hot = enc.transform(data_to_one_hot).toarray()
    data = data.drop(columns=cols_to_remove, inplace=False)
    new_data = np.concatenate((data_to_one_hot, data), axis=1)

    num_cols_per_categories = num_cols_per_categories + [1] * data.shape[1]

    return new_data, num_cols_per_categories


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

    layers_list = [ForwardLayer(n_features, 1, "identity")]  # simply a single linear layer

    return layers_list


def create_layers(n_features, n_layers, hidden_size, activation="leaky_relu"):

    layers_list = []

    if n_features > 1:
        layers_list.append(ForwardLayer(n_features, 1, "identity"))  # for dealing with categorical variables
    else:
        layers_list.append(ForwardLayer(n_features, hidden_size, activation))
        layers_list.append(BatchNormLayer(hidden_size))

        for _ in range(n_layers):
            layers_list.append(ForwardLayer(hidden_size, hidden_size, activation))
            layers_list.append(BatchNormLayer(hidden_size))

        layers_list.append(ForwardLayer(hidden_size, 1, 'identity'))

    return layers_list



