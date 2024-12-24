import numpy as np
import pickle
import os

from typing import Union, Any
from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import StandardScaler

from es_pca.utils import config_load, remove_outliers


CONFIG = config_load()


def if_empty_zero(array: np.array) -> np.array:
    if array.size == 0:
        array = 0

    return np.array(array)


def compute_variance_contribution(cov: np.array, comp: np.array, k: int, l: int, d: int) -> float:

    first_term = comp[k, l] ** 2 * cov[l, l]

    f1 = comp[k, :l] * comp[k, l] * cov[:l, l]
    f2 = comp[k, l+1:d] * comp[k, l] * cov[l+1:d, l]

    second_term = np.sum(if_empty_zero(f1)) + np.sum(if_empty_zero(f2))
    contribution = first_term + second_term

    return contribution


def get_contribs(cov: np.array, comp: int, p: int) -> np.array:
    arr_contrib = [[] for _ in range(p)]

    for pc_num in range(p):
        for feature_num in range(p):
            contrib = compute_variance_contribution(cov, comp, pc_num, feature_num, p)
            arr_contrib[pc_num].append(contrib)

    return np.array(arr_contrib)


def get_pca(run_index: int, data: np.array, training_mode: bool, save_pca_model: bool) -> tuple[PCA, np.array]:
    pca_type = CONFIG["pca_type"]
    pca_path = f"tmp_files/pca_model_{run_index}.pkl"

    # Initialize the PCA model based on the type specified
    if pca_type == "sparse":
        pca = SparsePCA(n_components=data.shape[1], alpha=CONFIG["alpha_reg_pca"])
    elif pca_type == "regular":
        pca = PCA(n_components=data.shape[1])
    else:
        raise ValueError(f"Invalid pca_type: {pca_type}. Expected one of ['sparse', 'regular'].")

    pca.fit(data)

    if save_pca_model and training_mode:
        with open(pca_path, "wb") as file:
            pickle.dump(pca, file)

    if not training_mode:
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model file '{pca_path}' not found.")

        with open(pca_path, "rb") as file:
            pca = pickle.load(file)

    pca_transformed_data = pca.transform(data)

    return pca, pca_transformed_data


def compute_fitness(run_index: int,
                    data_transformed: np.array,
                    training_mode: bool = True,
                    partial_contribution_objective: bool = False,
                    k: int = 1,
                    save_pca_model: bool = False) -> Union[list, Any]:

    if CONFIG["remove_outliers"] and training_mode:
        data_transformed = remove_outliers(data_transformed)

    scaler_path = f"tmp_files/scaler_model_{run_index}.pkl"
    scaler = StandardScaler()
    scaler.fit(data_transformed)

    if save_pca_model and training_mode:
        with open(scaler_path, "wb") as file:
            pickle.dump(scaler, file)

    if not training_mode:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"PCA model file '{scaler_path}' not found.")

        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)

    data_transformed = scaler.transform(data_transformed)

    pca_model, pca_transformed_data = get_pca(run_index,
                                              data_transformed,
                                              training_mode,
                                              save_pca_model)

    p = data_transformed.shape[1]
    cov_matrix = np.cov(np.transpose(data_transformed))

    variance_contrib = get_contribs(cov_matrix, pca_model.components_, p)

    total_variance_to_explain = np.sum(np.var(data_transformed, axis=0))

    if partial_contribution_objective:
        # this is using our novel objective function (breaking down the variance contribution per variable)
        score = np.sum(variance_contrib[:k], axis=0)/total_variance_to_explain
    else:
        # this is the regular PCA total explained variance
        score = [np.sum(variance_contrib[:k])/total_variance_to_explain]*p

    # numbers cannot be above 1 as they are standardized by the total amount of variance in the data
    # assert all(num < 1 for num in score), "Not all numbers are below 1"

    return score, pca_transformed_data, pca_model, scaler



