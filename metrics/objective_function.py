import copy
import numpy as np

from sklearn.decomposition import PCA
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


def get_pca(data):
    data = scale(data, axis=0)
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)

    return pca


def compute_fitness(data_transformed, partial_contribution_objective=False, k=1):
    pca_transformed = get_pca(copy.deepcopy(data_transformed))
    p = data_transformed.shape[1]

    if partial_contribution_objective:
        org_contrib = get_contribs(pca_transformed.get_covariance(), pca_transformed.components_, p, 0, p)
        score = list(np.sum(org_contrib[:k, :], axis=0))
    else:
        score = np.sum(pca_transformed.explained_variance_[:k])
        score = [score] * p

    return score
