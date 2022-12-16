import numpy as np
from typing import List
import copy

from layers.layers import Layer, ForwardLayer


class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def perturb(self, list_noise: List, sigma: float) -> List:

        copy_layers = copy.deepcopy(self.layers)

        counter = 0
        for layer in copy_layers:
            if isinstance(layer, ForwardLayer):
                w, b = copy.deepcopy(layer.get_weights())
                epsilon_w, epsilon_b = list_noise[counter]
                w_tmp = w + epsilon_w * sigma
                b_tmp = b + epsilon_b * sigma
                counter += 1

                layer.set_weights((w_tmp, b_tmp))

        return copy_layers

    def update_weights(self, update_to_add: np.ndarray):
        counter = 0
        for layer in self.layers:
            if isinstance(layer, ForwardLayer):
                layer_update = update_to_add[counter]
                w_update, b_update = layer_update[0], layer_update[1]
                w, b = layer.get_weights()
                w += w_update
                b += b_update
                layer.set_weights((w, b))

                counter += 1

        return self

    def predict(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x
