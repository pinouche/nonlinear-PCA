import numpy as np
from typing import List
import copy

from es_pca.layers.layers import Layer, ForwardLayer, BatchNormLayer


class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def perturb(self, list_noise: List, sigma: float) -> List:

        copy_layers = copy.deepcopy(self.layers)

        for l in range(len(copy_layers)):
            layer, noise = copy_layers[l], list_noise[l]
            if isinstance(layer, ForwardLayer):
                w, b = copy.deepcopy(layer.get_layer_weights())
                epsilon_w, epsilon_b = noise
                w_tmp = w + epsilon_w * sigma
                b_tmp = b + epsilon_b * sigma

                layer.set_weights((w_tmp, b_tmp))

        return copy_layers

    def update_weights(self, update_to_add: np.ndarray | list):

        for l in range(len(self.layers)):
            layer, layer_update = self.layers[l], update_to_add[l]
            if isinstance(layer, ForwardLayer):
                w_update, b_update = layer_update[0], layer_update[1]
                w, b = layer.get_layer_weights()
                w += w_update
                b += b_update
                layer.set_weights((w, b))

        return self

    def predict(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def get_weights(self):
        list_weights = []
        for layer in self.layers:
            w, b = layer.get_layer_weights()
            list_weights.append((w, b))

        return list_weights

    def get_noise_network(self):
        list_weights_noise = []
        for layer in self.layers:
            w, b = layer.get_layer_weights()
            if isinstance(layer, BatchNormLayer):  # we do not perturb the batch norm parameters.
                list_weights_noise.append((np.array([0]), np.array([0])))
            else:
                list_weights_noise.append((np.random.randn(*w.shape), np.random.randn(*b.shape)))

        return list_weights_noise

