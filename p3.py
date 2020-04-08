# neural network realisation from here
# https://habr.com/ru/post/271563/
# https://radioprog.ru/post/786
# https://radioprog.ru/post/780

import numpy as np

np.random.seed(1)


class Perceptron3:
    def __init__(self, inputs: int, hidden: int, outputs: int):
        # случайно инициализируем веса, в среднем - 0
        self.syn0 = 2 * np.random.random((inputs, hidden)) - 1
        self.syn1 = 2 * np.random.random((hidden, outputs)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def deriv_sigmoid(x):
        fx = Perceptron3.sigmoid(x)
        return fx * (1 - fx)

    def learn(self, inputs: [], outputs: [], epochs: int = 100000):
        inp = np.array(inputs)
        out = np.array(outputs)

        for i in range(epochs):

            # проходим вперёд по слоям 0, 1 и 2
            l0 = inp
            l1 = Perceptron3.sigmoid(np.dot(l0, self.syn0))
            l2 = Perceptron3.sigmoid(np.dot(l1, self.syn1))

            # как сильно мы ошиблись относительно нужной величины?
            l2_error = out - l2

            if (i % 10000) == 0: print("Error: ", str(np.mean(np.abs(l2_error))))

            # в какую сторону нужно двигаться?
            # если мы были уверены в предсказании, то сильно менять его не надо
            l2_delta = l2_error * Perceptron3.deriv_sigmoid(l2)

            # как сильно значения l1 влияют на ошибки в l2?
            l1_error = l2_delta.dot(self.syn1.T)

            # в каком направлении нужно двигаться, чтобы прийти к l1?
            # если мы были уверены в предсказании, то сильно менять его не надо
            l1_delta = l1_error * Perceptron3.deriv_sigmoid(l1)

            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += l0.T.dot(l1_delta)

    def feedforward(self, inputs: []):
        l0 = np.array(inputs)
        l1 = Perceptron3.sigmoid(np.dot(l0, self.syn0))
        l2 = Perceptron3.sigmoid(np.dot(l1, self.syn1))
        return l2


# набор входных данных
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# выходные данные
y = np.array([[0, 0, 1, 1]]).T

p3 = Perceptron3(3, 2, 1)
p3.learn(X, y)
print(p3.feedforward([0, 0, 1]))
print(p3.feedforward([0, 1, 1]))
print(p3.feedforward([1, 0, 1]))
print(p3.feedforward([0, 1, 0]))
