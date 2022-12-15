import numpy as np
from typing import List
import copy

from layers.layers import Layer

class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers[:]

    def perturb(self, list_noise: List, sigma: float) -> List:

        copy_layers = copy.deepcopy(self.layers)

        for layer_index in range(len(self.layers)):
            layer = copy_layers[layer_index]
            w, b = copy.deepcopy(layer.get_weights())
            epsilon_w, epsilon_b = list_noise[layer_index]
            w_tmp = w + epsilon_w * sigma
            b_tmp = b + epsilon_b * sigma

            layer.set_weights((w_tmp, b_tmp))

        return copy_layers

    def update_weights(self, update_to_add: np.ndarray) -> None:
        for i, layer in enumerate(self.layers):
            w_update, b_update = update_to_add[i]
            w, b = layer.get_weights()
            w += w_update
            b += b_update
            layer.set_weights((w, b))

    def predict(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x
