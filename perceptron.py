import numpy as np
from typing import List, Tuple

np.random.seed(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


class Perceptron:
    def __init__(self,
                 layer_1: int,
                 layer_2: int,
                 *other_layers: [int],
                 ):
        # задаём общий список нейронов
        layers = [layer_1, layer_2] + list(other_layers)
        # генерируем синапсы между нейронами
        self._synapses = [2 * np.random.random((layers[syn], layers[syn + 1])) - 1 for syn in range(len(layers) - 1)]

    def feedforward(self, inp: List[float]):
        li = np.array(inp)  # первый слой - входной
        for i in range(len(self._synapses)):  # проходим вперёд по слоям
            li = sigmoid(np.dot(li, self._synapses[i]))
        return li  # возвращаем последний слой

    def learn(self, inp, out,
              epochs: int = 100000,
              learn_coef: float = 1,
              err_print: bool = True,
              ):
        inp = np.array(inp)
        out = np.array(out)
        for epoch in range(epochs):
            self.shift_weights(inp, out, learn_coef, err_print and (epoch % 10000) == 0)

    def shift_weights(self,
                      inp: np.ndarray,
                      out: np.ndarray,
                      learn_coef: float = 1,
                      err_print: bool = True,
                      ):

        li = [inp]  # проходим вперёд по слоям 0, 1, 2 и т.д., сохраняем значения в li
        for i in range(len(self._synapses)):
            li.append(sigmoid(np.dot(li[i], self._synapses[i])))  # FEEDFORWARD

        # обратный проход
        out_error = out - li[-1]  # принт ошибки
        if err_print: print("Error: ", str(np.mean(np.abs(out_error))))
        # значения записываются в порядке от последнего к первому, хотя
        li_delta = [out_error * deriv_sigmoid(li[-1]) * learn_coef]  # 1-й элемент - уже смещённый результат

        for i in range(len(li) - 2, 0, -1):  # от последнего к первому не включая первый
            li_error = li_delta[0].dot(self._synapses[i].T)
            li_delta.insert(0, li_error * deriv_sigmoid(li[i]))

        # изменение весов с учётом коэфициента сдвига
        # print(li_delta)
        # print(li)
        for i in range(len(self._synapses)):
            # print(li[i + 1].T)
            # print(li_delta[i])
            # print(self._synapses[i])
            self._synapses[i] += li[i].T.dot(li_delta[i])


p3 = Perceptron(3, 2, 1)
p3.learn(np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]]),
         np.array([[0, 0, 1, 1]]).T
         )
print(p3.feedforward([0, 0, 1]))
print(p3.feedforward([0, 1, 1]))
print(p3.feedforward([1, 0, 1]))
print(p3.feedforward([1, 1, 0]))
