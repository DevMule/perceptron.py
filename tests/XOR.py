import perceptron
import numpy as np

np.random.seed(3)
# XOR check

ros = perceptron.Rosenblatt(2, 2, 1)
ros.learn(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [[0], [1], [1], [0]]
)
print(ros.feedforward([0, 0]))  # 0
print(ros.feedforward([0, 1]))  # 1
print(ros.feedforward([1, 0]))  # 1
print(ros.feedforward([1, 1]))  # 0

rum = perceptron.Rumelhart(2, 2, 1)
rum.learn(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [[0], [1], [1], [0]]
)
print(rum.feedforward([0, 0]))  # 0
print(rum.feedforward([0, 1]))  # 1
print(rum.feedforward([1, 0]))  # 1
print(rum.feedforward([1, 1]))  # 0
