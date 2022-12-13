import numpy as np
from typing import Tuple, List
from tqdm import tqdm
import math

from layers.layers import Layer
from metrics.objective_function import compute_fitness

# Define model
# class NeuralNetwork(nn.Module):
#    def __init__(self, n_layers, layer_size_list, input_size, output_size, n_components_to_keep):
#        super(NeuralNetwork, self).__init__()

#        self.layer_size_list = layer_size_list
#        self.n_layers = n_layers
#        self.output_size = output_size
#        self.input_size = input_size
#        self.k = n_components_to_keep

#        assert len(self.layer_size_list) == n_layers, f"`layer_size_list` should have length {n_layers}"

class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers[:]
        self.list_noise = []

    def update(self, sigma: float, lr: float, pop_size: int, x: np.ndarray, partial_contribution_objective: bool, num_components: int) -> None:

        for layer_index in range(len(self.layers)):
            w, b = self.layers[layer_index].get_weights()
            epsilon_w, epsilon_b = np.random.randn(w.shape), np.random.randn(b.shape)
            w += epsilon_w*sigma
            b += epsilon_b*sigma

            self.list_noise.append((w, b))

        F_obj = self.evaluate_model(x, partial_contribution_objective, num_components)

    def evaluate_model(self, x: np.ndarray, partial_contribution_objective: bool, num_components: int) -> [float]:
        """Returns objective value for given data"""

        output = self.predict(x)
        objective_value = compute_fitness(output, partial_contribution_objective, num_components)

        return objective_value

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            learning_rate: float, steps: int,
            batch_size: int, x_test: np.ndarray,
            y_test: np.ndarray) -> np.array:

        num_examples = x_train.shape[0]
        num_batches = math.ceil(num_examples / batch_size)
        metrics = np.zeros((steps // num_batches, 2))  # loss, accuracy
        random_index = np.linspace(0, num_examples - 1, num_examples).astype(int)
        for epoch in tqdm(range(steps // num_batches)):
            np.random.shuffle(random_index)
            x_train = x_train[random_index]
            y_train = y_train[random_index]
            for index_batch in range(0, num_examples, batch_size):
                mini_batch_x = x_train[index_batch: index_batch + batch_size]
                mini_batch_y = y_train[index_batch: index_batch + batch_size]
                self.loss_layer = CrossEntropyLoss(mini_batch_y)
                y_pred = self.predict(mini_batch_x, train=True)
                self.loss_layer.forward(y_pred)
                self.backward_propagation(y_pred)
                self.update_params(learning_rate)

            loss, accuracy = self.evaluate_model(x_test, y_test)
            metrics[epoch, :] = loss, accuracy

        return metrics

    def predict(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x
