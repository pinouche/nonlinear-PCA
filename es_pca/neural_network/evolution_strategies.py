import numpy as np
import copy
import os
import pickle
from typing import List, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import ndarray

from es_pca.metrics.objective_function import compute_fitness
from es_pca.neural_network.neural_network import NeuralNetwork
from es_pca.utils import create_scatter_plot, parse_arguments

args = parse_arguments()


class Solution:

    # here, we instantiate each neural network to be the same for each transformation (this could be easily custom).
    def __init__(self, network_list: List[NeuralNetwork]):
        self.networks = network_list

    def update(self, x_batch: np.ndarray, sigma: float, lr: float, pop_size: int,
               partial_contribution_objective: bool, num_components: int, run_index: int, epoch: int) -> None:

        # Accumulator for weighted noises per network and per layer.
        # Shape: list[num_networks][num_layers][2] with [weights_sum, bias_sum]
        accum_weighted_noise = [None] * len(self.networks)

        for _ in range(pop_size):
            # Generate noise for each network
            list_noise = [net.get_noise_network() for net in self.networks]
            x_transformed = self.predict(x_batch, sigma, list_noise, True)

            f_obj, _, _, _ = self.evaluate_model(
                x_transformed,
                partial_contribution_objective,
                num_components,
                True,
                False,
                run_index,
            )

            # Sanity check: one fitness value per network
            assert len(f_obj) == len(list_noise), (
                f"not the same length for list_noise {len(list_noise)} and f_obj {len(f_obj)}"
            )

            # Accumulate f_obj[i] * noise for each network and layer
            for net_i, noise_layers in enumerate(list_noise):
                if accum_weighted_noise[net_i] is None:
                    # Initialize with zeros using the shape of the first sampled noise
                    accum_weighted_noise[net_i] = [
                        [np.zeros_like(noise_layers[l][0]), np.zeros_like(noise_layers[l][1])]
                        for l in range(len(noise_layers))
                    ]

                coeff = f_obj[net_i]
                net_accum = accum_weighted_noise[net_i]
                for l, (eps_w, eps_b) in enumerate(noise_layers):
                    # eps_* can be arrays (ForwardLayer) or scalars (BatchNormLayer placeholder 0.0)
                    net_accum[l][0] += coeff * eps_w
                    net_accum[l][1] += coeff * eps_b

        # Scale by (lr/sigma)/pop_size to obtain the update step and apply to each network
        scale = (lr / sigma) / pop_size
        for i, net in enumerate(self.networks):
            updates = [[layer_sum[0] * scale, layer_sum[1] * scale] for layer_sum in accum_weighted_noise[i]]
            self.networks[i] = net.update_weights(updates)

        # save the neural network every 200 epochs
        epoch_period = 10
        if (epoch + 1) % epoch_period == 0:
            dataset_folder = "real_world_data"
            if args.dataset in ["circles", "spheres", "alternate_stripes"]:
                dataset_folder = "synthetic_data"

            # Insert PCA dimension subfolder to allow multiple dimensionalities per dataset
            saving_path = (f"results/datasets/{dataset_folder}/{args.dataset}/"
                           f"k={num_components}/"
                           f"activation={args.activation}/"
                           f"partial_contrib={str(partial_contribution_objective)}/{str(run_index)}/best_individual_epoch_{epoch}.p")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(saving_path), exist_ok=True)

            # Remove the existing file if it exists
            path_remove = (f"results/datasets/{dataset_folder}/{args.dataset}/"
                           f"k={num_components}/"
                           f"activation={args.activation}/"
                           f"partial_contrib={str(partial_contribution_objective)}/{str(run_index)}/best_individual_epoch_{epoch-epoch_period}.p")

            if os.path.exists(path_remove):
                os.remove(path_remove)

            # Save your object (replace `your_object` with the actual object you want to save)
            with open(saving_path, "wb") as f:
                pickle.dump(self.networks, f)

    def fit(self, x_train: np.ndarray, x_val: np.ndarray, classes: tuple[ndarray, ndarray], sigma: float,
            learning_rate: float, pop_size: int,
            partial_contribution_objective: bool,
            num_components: int,
            epochs: int,
            batch_size: int,
            early_stopping: int,
            train_indices: ndarray,
            val_indices: ndarray,
            run_index: int,
            latest_epoch: int = 0,
            result_list: list = None,
            verbose: bool = False) -> list[list[tuple[PCA, StandardScaler, Any, Any]]]:

        if result_list is None:
            result_list = []

        num_examples = x_train.shape[0]
        random_index = np.linspace(0, num_examples - 1, num_examples).astype(int)

        for epoch in range(latest_epoch, epochs):
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
                                run_index,
                                epoch)

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
                objective_train = objective_train[0] * len(objective_train) # rescale the objective value
                objective_val = objective_val[0] * len(objective_val)

            result_list.append([(pca_model, scaler, train_indices, val_indices, pca_transformed_val, pca_transformed_train),
                                (objective_train, objective_val)])
            print(f"the objective value for epoch {epoch} is: train {objective_train}, val {objective_val}")

            # save the neural network every 200 epochs
            epoch_period = 10
            if (epoch + 1) % epoch_period == 0:
                dataset_folder = "real_world_data"
                if args.dataset in ["circles", "spheres", "alternate_stripes"]:
                    dataset_folder = "synthetic_data"

                # Insert PCA dimension subfolder to allow multiple dimensionalities per dataset
                saving_path = (f"results/datasets/{dataset_folder}/{args.dataset}/"
                               f"k={num_components}/"
                               f"activation={args.activation}/"
                               f"partial_contrib={str(partial_contribution_objective)}/{str(run_index)}/"
                               f"results_list.p")

                # Ensure the directory exists
                os.makedirs(os.path.dirname(saving_path), exist_ok=True)

                # Save your object (replace `your_object` with the actual object you want to save)
                with open(saving_path, "wb") as f:
                    pickle.dump(result_list, f)

            if verbose and epoch % 1 == 0:
                self.plot((x_transformed_train, x_transformed_val),
                          (pca_transformed_train, pca_transformed_val),
                          classes)

        return result_list

    def plot(self, x_transformed: tuple[ndarray, ndarray],
             pca_transformed: tuple[ndarray, ndarray],
             classes: tuple[ndarray, ndarray]) -> None:
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
                       run_index: int) -> tuple[list[float], ndarray, PCA, StandardScaler]:

        score, pca_transformed_data, pca_model, scaler = compute_fitness(run_index,
                                                                         x_transformed,
                                                                         training_mode,
                                                                         partial_contribution_objective,
                                                                         num_components,
                                                                         save_pca_model)

        return score, pca_transformed_data, pca_model, scaler
