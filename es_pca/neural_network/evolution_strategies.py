import numpy as np
import copy
from typing import List, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from es_pca.metrics.objective_function import compute_fitness
from es_pca.neural_network.neural_network import NeuralNetwork
from es_pca.utils import convert_dic_to_list, create_scatter_plot


class Solution:

    # here, we instantiate each neural network to be the same for each transformation (this could be easily custom).
    def __init__(self, network_list: List[NeuralNetwork]):
        self.networks = network_list

    def update(self, x_batch: np.ndarray, sigma: float, lr: float, pop_size: int,
               partial_contribution_objective: bool, num_components: int, run_index: int) -> None:

        dict_weighted_noise = {}

        for p in range(pop_size):

            # get the noise. This generates noise for each of the neural network
            list_noise = [net.get_noise_network() for net in self.networks]
            x_transformed = self.predict(x_batch, sigma, list_noise, True)

            f_obj, _, _, _ = self.evaluate_model(x_transformed,
                                                 partial_contribution_objective,
                                                 num_components,
                                                 True,
                                                 False,
                                                 run_index)
            assert len(f_obj) == len(
                list_noise), f"not the same length for list_noise {len(list_noise)} and f_obj {len(f_obj)}"

            weighted_noise = [
                [
                    [f_obj[i] * arr for arr in tup]
                    for tup in inner_list
                ]
                for i, inner_list in enumerate(list_noise)
            ]

            for network_i in range(len(self.networks)):
                obj = f_obj[network_i]
                network = weighted_noise[network_i]
                if f"network_id_{network_i}" not in dict_weighted_noise:
                    dict_weighted_noise[f"network_id_{network_i}"] = {}

                for layer_id in range(len(network)):
                    layer = network[layer_id]
                    if layer_id not in dict_weighted_noise[f"network_id_{network_i}"]:
                        dict_weighted_noise[f"network_id_{network_i}"][layer_id] = []
                        dict_weighted_noise[f"network_id_{network_i}"][layer_id].append(layer[0] * obj)  # for weights
                        dict_weighted_noise[f"network_id_{network_i}"][layer_id].append(layer[1] * obj)  # for biases
                    else:
                        dict_weighted_noise[f"network_id_{network_i}"][layer_id][0] += layer[0] * obj
                        dict_weighted_noise[f"network_id_{network_i}"][layer_id][1] += layer[1] * obj

        # divide the values by the population size and multiply by (lr/sigma) to compute the update_step
        for outer_key, inner_dict in dict_weighted_noise.items():
            for key, value in inner_dict.items():
                dict_weighted_noise[outer_key][key][0] = (value[0] / pop_size) * (lr / sigma)
                dict_weighted_noise[outer_key][key][1] = (value[1] / pop_size) * (lr / sigma)

        # if partial_contribution_objective:
        self.networks = [self.networks[i].update_weights(convert_dic_to_list(dict_weighted_noise[f"network_id_{i}"]))
                         for i in range(len(self.networks))]

    def fit(self, x_train: np.ndarray, x_val: np.ndarray, classes: tuple[np.array, np.array], sigma: float,
            learning_rate: float, pop_size: int,
            partial_contribution_objective: bool,
            num_components: int,
            epochs: int,
            batch_size: int,
            early_stopping: int,
            train_indices: np.array,
            val_indices: np.array,
            run_index: int,
            verbose: bool = False) -> list[list[tuple[PCA, StandardScaler, Any, Any]]]:

        result_list = []
        num_examples = x_train.shape[0]
        random_index = np.linspace(0, num_examples - 1, num_examples).astype(int)

        best_objective_val = 0
        early_stopping_iterations = 0
        # x_transformed_train, x_transformed_val = None, None
        # pca_transformed_val, pca_transformed_train = None, None

        for epoch in range(epochs):
            print(f"COMPUTING FOR EPOCH {epoch}")
            np.random.shuffle(random_index)
            x_train_shuffled = copy.deepcopy(x_train[random_index])

            for index_batch in range(0, num_examples, batch_size):
                mini_batch_x = x_train_shuffled[index_batch: index_batch + batch_size]
                if mini_batch_x.shape[0] > x_train.shape[1]:  # n_components must be between 0 and min(n_samples, n_features) + small batches are too noisy.
                    self.update(mini_batch_x,
                                sigma,
                                learning_rate,
                                pop_size,
                                partial_contribution_objective,
                                num_components,
                                run_index)

            # evaluate objective at the end of the epoch on the training set.
            x_transformed_train = self.predict(x_train)
            objective_train, pca_transformed_train, pca_model, scaler = self.evaluate_model(x_transformed_train,
                                                                                            partial_contribution_objective,
                                                                                            num_components,
                                                                                            True,
                                                                                            True,
                                                                                            run_index)

            # evaluate objective at the end of the epoch on the validation set
            x_transformed_val = self.predict(x_val)
            objective_val, pca_transformed_val, pca_model, scaler = self.evaluate_model(
                x_transformed_val,
                partial_contribution_objective,
                num_components,
                False,
                False,
                run_index)

            # for partial contribution = True, each element is the explained variance for each variable.
            # for partial contribution = False, each element of the list is the (duplicated) total variance -> do not sum.
            if partial_contribution_objective:
                # the standardizing is done on training data so the total var to explain is the number of variables
                objective_train = np.sum(objective_train)
                objective_val = np.sum(objective_val)
            else:
                objective_train = objective_train[0]
                objective_val = objective_val[0]

            result_list.append([(pca_model, scaler, train_indices, val_indices),
                                (objective_train, objective_val)])
            print(f"the objective value for epoch {epoch} is: train {objective_train}, val {objective_val}")

            # implement early stopping
            if objective_val > best_objective_val:
                best_objective_val = objective_val
                early_stopping_iterations = 0
            else:
                early_stopping_iterations += 1
                if early_stopping_iterations >= early_stopping:
                    break

            if verbose and epoch % 1 == 0:
                self.plot((x_transformed_train, x_transformed_val),
                          (pca_transformed_train, pca_transformed_val),
                          classes)

        return result_list

    def plot(self, x_transformed: tuple[np.array, np.array],
             pca_transformed: tuple[np.array, np.array],
             classes: tuple[np.array, np.array]) -> None:
        create_scatter_plot(x_transformed, pca_transformed, classes)

    def predict(self, x: np.ndarray, sigma: float = None, list_noise: list = None, train: bool = True) -> np.ndarray:
        x_transformed = np.empty((x.shape[0], len(self.networks)))

        input_index = 0
        for i, network in enumerate(self.networks):
            # here, we get the dimension of the inputs to know which variables of the dataset it corresponds to.
            input_dim = network.layers[0].weights.shape[0]  # we can ignore this "unresolved attribute"

            perturbed_network = copy.deepcopy(network)
            if list_noise:
                perturbed_layers = network.perturb(list_noise[i], sigma)
                perturbed_network = NeuralNetwork(perturbed_layers)

            output_perturbed_network = perturbed_network.predict(x[:, input_index:input_index + input_dim], train)
            x_transformed[:, i] = np.squeeze(output_perturbed_network)
            input_index += input_dim

        return x_transformed

    def evaluate_model(self,
                       x_transformed: np.ndarray,
                       partial_contribution_objective: bool,
                       num_components: int,
                       training_mode: bool,
                       save_pca_model: bool,
                       run_index: int) -> tuple[list[float], np.array, PCA, StandardScaler]:

        score, pca_transformed_data, pca_model, scaler = compute_fitness(run_index,
                                                                         x_transformed,
                                                                         training_mode,
                                                                         partial_contribution_objective,
                                                                         num_components,
                                                                         save_pca_model)

        return score, pca_transformed_data, pca_model, scaler
