import numpy as np
import copy
from typing import List, Tuple

from es_pca.metrics.objective_function import compute_fitness
from es_pca.neural_network.neural_network import NeuralNetwork


class Solution:

    # here, we instantiate each neural network to be the same for each transformation (this could be easily custom).
    def __init__(self, network_list: List[NeuralNetwork]):
        self.networks = network_list

    def update(self, x_batch: np.ndarray, sigma: float, lr: float, pop_size: int, pca_reg: float,
               partial_contribution_objective: bool, num_components: int) -> None:

        list_weighted_noise = []

        for p in range(pop_size):

            x_transformed = np.empty((x_batch.shape[0], len(self.networks)))
            # get the noise
            list_noise = [net.get_noise_network() for net in self.networks]

            input_index = 0
            for i, network in enumerate(self.networks):

                input_dim = network.layers[0].weights.shape[0]

                perturbed_layers = network.perturb(list_noise[i], sigma)
                perturbed_network = NeuralNetwork(perturbed_layers)
                output_perturbed_network = perturbed_network.predict(x_batch[:, input_index:input_index+input_dim])
                x_transformed[:, i] = np.squeeze(output_perturbed_network)

                input_index += input_dim

            f_obj = self.evaluate_model(x_transformed, pca_reg, partial_contribution_objective, num_components)
            assert len(f_obj) == len(list_noise), f"not the same length for list_noise {len(list_noise)} and f_obj {len(f_obj)}"
            weighted_noise = [f_obj[i]*np.array(list_noise[i]) for i in range(len(f_obj))]
            list_weighted_noise.append(weighted_noise)

        gradient_estimate = np.mean(np.array(list_weighted_noise), axis=0)
        update_step = [grad*(lr/sigma) for grad in gradient_estimate]

        if partial_contribution_objective:
            self.networks = [self.networks[i].update_weights(update_step[i]) for i in range(len(self.networks))]
        else:
            self.networks = [net.update_weights(update_step[0]) for net in self.networks]

    def fit(self, x_train: np.ndarray, x_val: np.ndarray, sigma: float, learning_rate: float, pop_size: int, pca_reg: float,
            partial_contribution_objective: bool, num_components: int, epochs: int, batch_size: int, early_stopping: int) -> Tuple:

        objective_list = []
        num_examples = x_train.shape[0]
        random_index = np.linspace(0, num_examples - 1, num_examples).astype(int)

        best_objective_val = 0
        early_stopping_iterations = 0
        x_transformed_train = None
        x_transformed_val = None
        for epoch in range(epochs):
            print(f"COMPUTING FOR EPOCH {epoch}")
            np.random.shuffle(random_index)
            x_train_shuffled = copy.deepcopy(x_train[random_index])

            for index_batch in range(0, num_examples, batch_size):
                mini_batch_x = x_train_shuffled[index_batch: index_batch + batch_size]
                self.update(mini_batch_x, sigma, learning_rate, pop_size, pca_reg, partial_contribution_objective, num_components)
                print("DONE BATCH")

            # evaluate objective at the end of the epoch on the training set
            x_transformed_train = self.predict(x_train, True)
            objective_train = self.evaluate_model(x_transformed_train, pca_reg, False, num_components)[0]

            # evaluate objective at the end of the epoch on the validation set
            x_transformed_val = self.predict(x_val, False)
            objective_val = self.evaluate_model(x_transformed_val, pca_reg, False, num_components)[0]

            objective_list.append((objective_train, objective_val))
            print(f"the objective value for epoch {epoch} is {objective_train, objective_val}")

            # implement early stopping
            if objective_val > best_objective_val:
                best_objective_val = objective_val
                early_stopping_iterations = 0
            else:
                early_stopping_iterations += 1
                if early_stopping_iterations >= early_stopping:
                    break

        return objective_list, (x_transformed_train, x_transformed_val)

    def predict(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        x_transformed = np.empty(x.shape)
        for i, network in enumerate(self.networks):
            output_perturbed_network = network.predict(x[:, i], train)
            x_transformed[:, i] = np.squeeze(output_perturbed_network)

        return x_transformed

    def evaluate_model(self, x_transformed: np.ndarray, pca_reg: float, partial_contribution_objective: bool, num_components: int) -> [float]:

        objective_value = compute_fitness(x_transformed, pca_reg, partial_contribution_objective, num_components)

        return objective_value

