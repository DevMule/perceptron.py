import numpy as np
import perceptron

# print(perceptron.__doc__)

# XOR check
p3 = perceptron.Perceptron(2, 4, 1)
p3.learn(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [[0], [1], [1], [0]],
    learn_rate=1
)
print(p3.feedforward([0, 0]))  # 0
print(p3.feedforward([0, 1]))  # 1
print(p3.feedforward([1, 0]))  # 1
print(p3.feedforward([1, 1]))  # 0
