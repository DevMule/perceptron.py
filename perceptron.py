import numpy as np
from typing import List, Union

np.random.seed(1)


def sigmoid(x: Union[int, float, np.ndarray]):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x: Union[int, float, np.ndarray]):
    fx = sigmoid(x)
    return fx * (1 - fx)


class Perceptron:
    def __init__(self, layer_1: int, layer_2: int, *other_layers: [int]):
        layers = [layer_1, layer_2] + list(other_layers)
        self._synapses = [2 * np.random.random((layers[syn], layers[syn + 1])) - 1 for syn in range(len(layers) - 1)]

    def feedforward(self, inp: Union[List[float], np.ndarray]):
        li = np.array(inp)
        for i in range(len(self._synapses)): li = sigmoid(np.dot(li, self._synapses[i]))
        return li

    def learn(self, inp, out, epochs: int = 100000, learn_coef: Union[float, int] = 1, err_print: bool = True):
        for epoch in range(epochs):
            self.shift_weights(np.array(inp), np.array(out), learn_coef, err_print and (epoch % 10000) == 0)

    def shift_weights(self, inp: np.ndarray, out: np.ndarray, learn_coef: float = 1, err_print: bool = True):
        li = [inp]
        for i in range(len(self._synapses)): li.append(sigmoid(np.dot(li[i], self._synapses[i])))
        d_li = [(out - li[-1]) * deriv_sigmoid(li[-1]) * learn_coef]
        for i in range(len(li) - 2, 0, -1): d_li.insert(0, d_li[0].dot(self._synapses[i].T) * deriv_sigmoid(li[i]))
        for i in range(len(self._synapses)): self._synapses[i] += li[i].T.dot(d_li[i])
        if err_print: print("Error: ", str(np.mean(np.abs(out - self.feedforward(inp)))))


if __name__ == "__main__":
    p3 = Perceptron(3, 5, 8, 1)
    p3.learn(np.array([[0, 0, 1],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]]),
             np.array([[0, 0, 1, 1]]).T
             )
    print(p3.feedforward([0, 0, 1]))  # 0
    print(p3.feedforward([0, 1, 1]))  # 0
    print(p3.feedforward([1, 0, 1]))  # 1
    print(p3.feedforward([1, 1, 0]))  # 1
