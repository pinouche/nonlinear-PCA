import copy
import numpy as np

from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import scale


def if_empty_zero(array):
    if array.size == 0:
        array = 0

    return array


def compute_variance_contribution(cov, comp, k, l, d):

    first_term = comp[k, l] ** 2 * cov[l, l]

    f1 = comp[k, :l] * comp[k, l] * cov[:l, l]
    f2 = comp[k, l+1:d] * comp[k, l] * cov[l+1:d, l]

    second_term = np.sum(if_empty_zero(f1)) + np.sum(if_empty_zero(f2))
    contribution = first_term + second_term

    return contribution


def get_contribs(cov, comp, k, start, end):
    arr_contrib = [[] for _ in range(k)]

    for pc_num in range(k):
        for feature_num in range(start, end):
            contrib = compute_variance_contribution(cov, comp, pc_num, feature_num, k)
            arr_contrib[pc_num].append(contrib)

    return np.array(arr_contrib)


def get_pca(data, alpha=0.01):
    data = scale(data, axis=0)

    if alpha > 0:
        pca = SparsePCA(n_components=data.shape[1], alpha=alpha)
    else:
        pca = PCA(n_components=data.shape[1])

    pca.fit(data)

    return pca


def compute_fitness(data_transformed, alpha, partial_contribution_objective=False, k=1):
    data_transformed = scale(data_transformed, axis=0)
    pca_transformed = get_pca(copy.deepcopy(data_transformed), alpha)
    p = data_transformed.shape[1]
    cov_matrix = np.cov(np.transpose(data_transformed))

    variance_contrib = get_contribs(cov_matrix, pca_transformed.components_, p, 0, p)

    if partial_contribution_objective:
        score = np.sum(variance_contrib[:k], axis=0)
    else:
        score = [np.sum(variance_contrib[:k])]*p

    return score
