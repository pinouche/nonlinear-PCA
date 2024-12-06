import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class NonLinearities:

    def __init__(self, activation: str):
        self.activation = activation

    def compute_output(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            x[x < 0] = 0
        elif self.activation == 'leaky_relu':
            x = np.where(x > 0, x, x * 0.01)
        elif self.activation == 'sigmoid':
            x = 1 / (1 + np.exp(x * -1))
        elif self.activation == 'selu':
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            x = np.where(x > 0, x, alpha * (np.exp(x) - 1)) * scale
        elif self.activation == 'cos':
            x = np.cos(x)
        elif self.activation == 'identity':
            x = x
        else:
            raise ValueError('Invalid activation {}'.format(self.activation))

        return x


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def get_layer_weights(self) -> Tuple:
        pass


class ForwardLayer(Layer):

    def __init__(self, input_dim: int, output_dim: int, activation: str = 'leaky_relu', init_mode: str = "normal"):
        self.activation_fn = NonLinearities(activation)

        if init_mode == "normal":
            self.weights = np.random.randn(input_dim, output_dim)
        elif init_mode == "identity":
            self.weights = create_rectangular_identity(input_dim, output_dim)
        else:
            raise ValueError(f"The init_mode {init_mode} is not accepted.")

        self.biases = np.zeros((1, output_dim))

    def set_weights(self, params: Tuple) -> None:
        self.weights, self.biases = params

    def get_layer_weights(self) -> Tuple:
        return self.weights, self.biases

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if len(x.shape) == 1:
            x = np.expand_dims(x, 1)
        x = np.matmul(x, self.weights) + self.biases
        x = self.activation_fn.compute_output(x)

        return x


# this is a monotonic layer from the paper: Monotonic Networks by Joseph Sill
class MonotonicForwardLayer(Layer):

    def __init__(self, input_dim: int, output_dim: int, monotonicity: str = 'increasing'):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.zeros((1, output_dim))
        self.monotonicity = monotonicity

    def set_weights(self, params: Tuple) -> None:
        self.weights, self.biases = params

    def get_layer_weights(self) -> Tuple:
        return self.weights, self.biases

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if len(x.shape) == 1:
            x = np.expand_dims(x, 1)

        if self.monotonicity == "increasing":
            z = np.exp(self.weights)
        elif self.monotonicity == "decreasing":
            z = -1 * np.exp(self.weights)
        else:
            raise ValueError('Invalid motonicity {}'.format(self.monotonicity))

        x = np.matmul(x, z) + self.biases

        return x


class BatchNormLayer(Layer):

    def __init__(self, dims: int) -> None:
        self.gamma = np.ones((1, dims), dtype="float32")
        self.bias = np.zeros((1, dims), dtype="float32")

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9

        self.epsilon = 10 ** -3

    def update_running_variables(self) -> None:
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Var running averages should be "
                             "initilizaded at the same time")
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + \
                                  (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + \
                                 (1. - gamma) * self.var_x

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.num_examples = x.shape[0]
        if train:
            self.mean_x = np.mean(x, axis=0, keepdims=True)
            self.var_x = np.mean((x - self.mean_x) ** 2, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += self.epsilon
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = x - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x

        return self.gamma * self.standard_x + self.bias

    def get_layer_weights(self) -> Tuple:
        return np.array([0]), np.array([0])


def create_rectangular_identity(p: int, h: int) -> np.array:

    if h == 1:
        result = np.ones((p, h), dtype=float)
    else:
        result = np.zeros((p, h), dtype=float)
        result[:p, :p] = np.eye(p)

    return result
