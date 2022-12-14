import numpy as np
from typing import List

from layers.layers import Layer


class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers[:]
        self.list_noise = []

    def predict(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x
