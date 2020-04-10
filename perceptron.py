import numpy as np
from typing import List, Union

np.random.seed(1)


def sigmoid(x: Union[int, float, np.ndarray]) -> float:
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x: Union[int, float, np.ndarray]) -> float:
    fx = sigmoid(x)
    return fx * (1 - fx)


class Perceptron:
    def __init__(self, layer_1: int, layer_2: int, *other_layers: int):
        # список слоёв и их размеров
        layers = [layer_1, layer_2] + list(other_layers)
        # генерируются синапсы между уровнями нейронов, которые представляют из себя массив весов
        # Аij, где i и j - размеры соединяемых слоёв, каждый вес - вес между каждым из нейронов
        self._synapses = []
        self._biases = []
        for syn in range(len(layers) - 1):
            # каждый нейрон связываем с каждым нейроном предыдущего слоя случайным значением
            self._synapses.append(2 * np.random.random((layers[syn], layers[syn + 1])) - 1)
            # для каждого нейрона для всех слоёв кроме 1-го задаём случайные сдвиги
            self._biases.append(2 * np.random.random(layers[syn + 1]) - 1)

    def feedforward(self, inp: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Функция прямого распространения.
        inp - список входных значений
        Возвращает выходные сигналы, соответствющие данному вводному набору.
        """
        li = np.array(inp)  # входной слой
        for i in range(len(self._synapses)):  # для каждого последующего
            # с использованием предыдущего слоя рассчитываются значения следующего
            li = sigmoid(np.dot(li, self._synapses[i]) + self._biases[i])
        return li  # последний слой - выходной

    def learn(self,
              inp: Union[List[List[float]], np.ndarray],
              out: Union[List[List[float]], np.ndarray],
              epochs: int = 100000,
              learn_coef: Union[float, int] = 1,
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

            # прямой ход - рассчёт значений слоёв
            li = [inp]  # список рассчитанных значений для каждого слоя для каждого входного значения начиная с входного
            for i in range(len(self._synapses)):
                # т.к. ВСЕ входные значения хранятся в матрице,
                # для каждого слоя рассчитываются значения для каждого из входных значений
                li.append(sigmoid(np.dot(li[i], self._synapses[i]) + self._biases[i]))

            # обратный ход - рассчёт необходимых смещений для синапсов
            # для последнего слоя рассчитывается ошибка = разница между рассчётными и необходимыми выходными сигналами
            # из ошибки рассчитывется дельта ошибки - величина на которую необходимо совершить смещение
            #      [(   ошибка   ) * производная значения ] = смещение, необходимое для минимизации ошибки в будущем
            d_li = [(out - li[-1]) * deriv_sigmoid(li[-1])]  # необходимые смещения для каждого слоя
            for i in range(len(li) - 2, 0, -1):  # обратный проход не включая первый и последний слои
                # используя рассчитанное значение сдвига следующего слоя, рассчитывается ошибка для предыдущего слоя
                li_err = d_li[0].dot(self._synapses[i].T)
                # аналогично из ошибки рассчитывется величина на которую необходимо совершить смещение
                d_li.insert(0, li_err * deriv_sigmoid(li[i]))

            # применение смещений для синапсов и сдвигов
            for i in range(len(self._synapses)):
                # смещения применяются к синапсам
                self._synapses[i] += li[i].T.dot(d_li[i]) * learn_coef
                # смещения рассчитываются и применяются к сдвигам
                # т.к. d_li представляет из себя смещение для каждого нейрона для каждого из заданных пар значений
                # то для каждого нейрона находится среднее значение смещения
                self._biases[i] += np.mean(d_li[i], axis=0) * learn_coef

            # если есть необходимость, в консоль выводится ошибка
            if err_print and (epoch % err_print_frequency) == 0:
                print("Error: ", str(np.mean(np.abs(out - self.feedforward(inp)))))


if __name__ == "__main__":
    p3 = Perceptron(3, 2, 1)
    p3.learn(
        np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]]),
        np.array([[0],
                  [0],
                  [1],
                  [1],
                  [1]])
    )
    print(p3.feedforward([0, 1, 1]))  # 0
    print(p3.feedforward([0, 0, 1]))  # 0
    print(p3.feedforward([1, 1, 0]))  # 1
    print(p3.feedforward([1, 0, 1]))  # 1 ?
