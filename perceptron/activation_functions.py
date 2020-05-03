import numpy as np
from typing import Union


def sigmoid(x: Union[int, float, np.ndarray]) -> float:
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x: Union[int, float, np.ndarray]) -> float:
    fx = sigmoid(x)
    return fx * (1 - fx)
