import copy
import numpy as np

from typing import Union, Any

from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import scale

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


def get_pca(data: np.array, alpha: float = 0.01) -> tuple[PCA, np.array]:
    pca_type = CONFIG["pca_type"]
    data = scale(data, axis=0)

    if pca_type == "sparse":
        pca = SparsePCA(n_components=data.shape[1], alpha=CONFIG["alpha_reg_pca"])
    elif pca_type == "regular":
        pca = PCA(n_components=data.shape[1])
    elif pca_type == "robust":
        pca = Rpca(data)
        L, S = pca.fit(max_iter=1000, iter_print=100)
        data = L
        pca = PCA(n_components=data.shape[1])
    else:
        raise ValueError(f"pca_type is not valid, expected to be one of [sparse, robust, regular], got {pca_type}")

    pca.fit(data)
    pca_transformed_data = pca.transform(data)

    return pca, pca_transformed_data


def compute_fitness(data_transformed: np.array,
                    evalutation_mode: bool,
                    alpha: float = 0.0,
                    partial_contribution_objective: bool = False,
                    k: int = 1) -> Union[list, Any]:

    data_transformed = scale(data_transformed, axis=0)
    if CONFIG["remove_outliers"] and evalutation_mode:
        data_transformed = remove_outliers(data_transformed)
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


class Rpca:

    """Implementation taken from https://github.com/dganguli/robust-pca"""

    def __init__(self, D, mu=None, lmbda=None):
        self.L = None
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.epsilon = 10e-5

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / ((4 * np.linalg.norm(self.D, ord=1))+self.epsilon)

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        # this loop implements the principal component pursuit (PCP) algorithm
        # located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            # if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
            #     print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk


