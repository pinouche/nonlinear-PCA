import copy
import numpy as np

from typing import Union, Any

from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import scale


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


def get_pca(data: np.array, alpha: float = 0.01) -> tuple[PCA, np.array]:
    data = scale(data, axis=0)

    if alpha > 0:
        pca = SparsePCA(n_components=data.shape[1], alpha=alpha)
    else:
        pca = PCA(n_components=data.shape[1])

    pca.fit(data)
    pca_transformed_data = pca.transform(data)

    return pca, pca_transformed_data


def compute_fitness(data_transformed: np.array, alpha: float,
                    partial_contribution_objective: bool = False, k: int = 1) -> Union[list, Any]:
    data_transformed = scale(data_transformed, axis=0)
    pca_model, pca_transformed_data = get_pca(copy.deepcopy(data_transformed), alpha)
    p = data_transformed.shape[1]
    cov_matrix = np.cov(np.transpose(data_transformed))

    variance_contrib = get_contribs(cov_matrix, pca_model.components_, p)

    if partial_contribution_objective:
        score = np.sum(variance_contrib[:k], axis=0)
    else:
        score = [np.sum(variance_contrib[:k])]*p
    #
    # print(f"the variance contribution is: {score}")

    return score, pca_transformed_data


