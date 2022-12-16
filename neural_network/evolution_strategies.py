import numpy as np
from typing import List, Tuple

from metrics.objective_function import compute_fitness
from neural_network.neural_network import NeuralNetwork
from layers.layers import ForwardLayer


class Solution:

    # here, we instantiate each neural network to be the same for each transformation, but this could be easily customed.
    def __init__(self, network_list: List[NeuralNetwork]):
        self.networks = network_list

    def update(self, x_batch: np.ndarray, sigma: float, lr: float, pop_size: int, pca_reg: float, partial_contribution_objective: bool,
               num_components: int) -> None:

        # here, we assume all networks are the same topology
        list_weights_shape = [(layer.get_weights()[0].shape, layer.get_weights()[1].shape) for layer in self.networks[0].layers if isinstance(layer, ForwardLayer)]
        list_weighted_noise = []

        for p in range(pop_size):

            x_transformed = np.empty(x_batch.shape)
            # at each population iteration, we add the same noise to all of the networks (no loss of generality).
            list_noise = [(np.random.randn(*tup[0]), np.random.randn(*tup[1])) for tup in list_weights_shape]

            for i, network in enumerate(self.networks):

                perturbed_layers = network.perturb(list_noise, sigma)
                perturbed_network = NeuralNetwork(perturbed_layers)
                output_perturbed_network = perturbed_network.predict(x_batch[:, i])
                x_transformed[:, i] = np.squeeze(output_perturbed_network)

            f_obj = self.evaluate_model(x_transformed, pca_reg, partial_contribution_objective, num_components)
            weighted_noise = f_obj*np.array(list_noise)
            list_weighted_noise.append(weighted_noise)

        gradient_estimate = np.mean(np.array(list_weighted_noise), axis=0)
        update_step = gradient_estimate*(lr/sigma)
        self.networks = [net.update_weights(update_step) for net in self.networks]

    def fit(self, x_train: np.ndarray, sigma: float, learning_rate: float, pop_size: int, pca_reg: float, partial_contribution_objective: bool, num_components: int,
            epochs: int, batch_size: int, train: bool) -> Tuple:

        objective_list = []
        num_examples = x_train.shape[0]
        random_index = np.linspace(0, num_examples - 1, num_examples).astype(int)

        for epoch in range(epochs):
            print(f"COMPUTING FOR EPOCH {epoch}")
            np.random.shuffle(random_index)
            x_train = x_train[random_index]

            for index_batch in range(0, num_examples, batch_size):
                mini_batch_x = x_train[index_batch: index_batch + batch_size]
                self.update(mini_batch_x, sigma, learning_rate, pop_size, pca_reg, partial_contribution_objective, num_components)
                print("DONE BATCH")

            # evaluate objective at the end of the epoch
            x_transformed = self.predict(x_train, train)
            objective = self.evaluate_model(x_transformed, pca_reg, partial_contribution_objective, num_components)
            objective_list.append(objective)

            print(f"the objective value for epoch {epoch} is {objective}")

        return objective_list, x_transformed

    def predict(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        x_transformed = np.empty(x.shape)
        for i, network in enumerate(self.networks):
            output_perturbed_network = network.predict(x[:, i], train)
            x_transformed[:, i] = np.squeeze(output_perturbed_network)

        return x_transformed

    def evaluate_model(self, x_transformed: np.ndarray, pca_reg: float, partial_contribution_objective: bool, num_components: int) -> [float]:

        objective_value = compute_fitness(x_transformed, pca_reg, partial_contribution_objective, num_components)

        return objective_value

