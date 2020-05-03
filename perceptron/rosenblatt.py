import numpy as np
from typing import List, Union
from .activation_functions import sigmoid, deriv_sigmoid


class Rosenblatt:
    def __init__(self, l_inp: int = 2, l_hid: int = 3, l_out: int = 1):
        self.ih_synapses = 2 * np.random.random((l_inp, l_hid)) - 1
        self.ho_synapses = 2 * np.random.random((l_hid, l_out)) - 1
        self.h_biases = 2 * np.random.random(l_hid) - 1
        self.o_biases = 2 * np.random.random(l_out) - 1

    def feedforward(self, inp: Union[List[float], np.ndarray]) -> List[float]:
        inp = np.array(inp)
        h_neurons = sigmoid(np.dot(inp, self.ih_synapses) + self.h_biases)
        return sigmoid(np.dot(h_neurons, self.ho_synapses) + self.o_biases).tolist()

    def learn(self,
              inp: List[List[float]],
              out: List[List[float]],
              epochs: int = 100000,
              learning_rate: Union[float, int] = .1,
              err_print: bool = True,
              err_print_frequency: int = 10000,
              ):
        for epoch in range(epochs):
            # рассчитать значения слоёв прямым проходом
            li = np.array(inp)
            lh = sigmoid(np.dot(li, self.ih_synapses) + self.h_biases)
            lo = sigmoid(np.dot(lh, self.ho_synapses) + self.o_biases)

            # обучение проходит по следующей формуле:
            # d_Wij = learning_rate * Em * deriv_sigmoid(m) * L.T , где
            # d_Wij = величина, на которую необходимо сместить вес между нейронами i слоя L и j слоя M
            # learning_rate = коэффициент обучения
            # Em = ошибка значений нейронов слоя M, deriv_sigmoid(m) = производная функция в рассчитанных координатах
            # M * deriv_sigmoid(m) = градиент слоя M
            # L.T = транспонированная матрица рассчитанных значений нейронов предыдущего слоя L/
            err_o = np.array(out) - lo  # рассчёт значения ошибки выходного слоя
            d_lo = err_o * deriv_sigmoid(lo)  # рассчёт градиента выходного слоя

            err_h = d_lo.dot(self.ho_synapses.T)  # рассчёт значения ошибки скрытого слоя
            d_lh = err_h * deriv_sigmoid(lh)  # рассчёт градиента скрытого слоя

            self.ho_synapses += lh.T.dot(d_lo) * learning_rate  # сдвиг весов на нужное значение
            self.ih_synapses += li.T.dot(d_lh) * learning_rate
            self.o_biases += np.mean(d_lo, axis=0) * learning_rate
            self.h_biases += np.mean(d_lh, axis=0) * learning_rate
            if err_print and (epoch % err_print_frequency) == 0:
                print("Error: ", str(np.mean(np.abs(out - lo))))
