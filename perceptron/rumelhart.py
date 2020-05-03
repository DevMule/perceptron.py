import numpy as np
from typing import List, Union
from .activation_functions import sigmoid, deriv_sigmoid


class Rumelhart:
    def __init__(self, layer_1: int, layer_2: int, *other_layers: int):
        # список слоёв и их размеров
        layers = [layer_1, layer_2] + list(other_layers)
        self._synapses = []
        self._biases = []
        for i in range(len(layers) - 1):
            # генерируются синапсы между уровнями нейронов, которые представляют из себя массив весов
            # матрица Аij, где i и j - размеры соединяемых слоёв, каждый вес - вес между каждым из нейронов
            # каждый нейрон связываем с каждым нейроном предыдущего слоя случайным значением
            self._synapses.append((2 * np.random.random((layers[i], layers[i + 1])) - 1))
            # генерируются сдвиги для каждого нейрона всех слоёв кроме первого
            # вектор Bi, где i - размер слоя. для каждого нейрона для всех слоёв кроме 1-го задаются случайные сдвиги
            self._biases.append((2 * np.random.random(layers[i + 1]) - 1))

    def feedforward(self, inp: Union[List[float], np.ndarray]) -> List[float]:
        """
        Функция прямого распространения.
        inp - список входных значений
        Возвращает выходные сигналы, соответствющие данному вводному набору.
        """
        li = np.array(inp)  # рассчитанный слой
        for i in range(len(self._synapses)):  # для каждого последующего
            # с использованием предыдущего слоя рассчитываются значения следующего
            li = sigmoid(np.dot(li, self._synapses[i]) + self._biases[i])
        return li.tolist()  # последний слой - выходной

    def learn(self,
              inp: List[List[float]],
              out: List[List[float]],
              epochs: int = 100000,
              learning_rate: Union[float, int] = .1,
              err_print: bool = True,
              err_print_frequency: int = 10000,
              ):
        """
        Функция обучения.
        inp, out - списки входных и выходных значений соответственно. Количество элементов в списках должно совпадать.
        epochs (int) - количество проходов по спискам
        learn_coef (int) - коэфициент сдига весов при одном проходе
        err_print (Bool) - флаг, разрешение на вывод ошибки
        Изменяет веса для сети в соответствии с парами данных.
        """
        inp = np.array(inp)
        out = np.array(out)
        for epoch in range(epochs):

            # прямой ход - расчёт сумм и значений нейронов
            Si = [inp]  # суммы нейронов
            Xi = [sigmoid(Si[-1])]  # значения нейронов
            for i in range(len(self._synapses)):
                Si.append(np.dot(Xi[i], self._synapses[i]) + self._biases[i])
                Xi.append(sigmoid(Si[-1]))

            # перенос ошибки на скрытые слои
            Ei = [out - Xi[-1]]  # ошибки для всех кроме входного слоёв сети
            for i in range(len(Xi) - 2, 0, -1):  # проход по всем скрытым слоям
                # Ei = Ei+1 * Wi
                Ei.insert(0, Ei[0].dot(self._synapses[i].T))

            # расчёт дельт весов синапсов
            for i in range(len(self._synapses)):
                # E * deriv_sigmoid(Si)  скалярно значения между собой
                grad = np.multiply(Ei[i], deriv_sigmoid(Si[i + 1]))
                dw = np.dot(grad.T, Xi[i]).T
                self._synapses[i] += dw * learning_rate
                self._biases[i] += np.mean(dw, axis=0) * learning_rate

            # вывод ошибки при необходимости
            if err_print and (epoch % err_print_frequency) == 0:
                print("Error: ", str(np.mean(np.abs(Ei[-1]))))
