import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        pass


class ParamLayer(Layer, ABC):

    @abstractmethod
    def apply_gradients(self, learning_rate: float) -> None:
        pass

