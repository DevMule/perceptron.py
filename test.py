import perceptron
import numpy as np

np.random.seed(1)
# XOR check
p3 = perceptron.Perceptron(2, 2, 1)
p3.learn(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [[0], [1], [1], [0]],
    epochs=200000
)
print(p3.feedforward([0, 0]))  # 0
print(p3.feedforward([0, 1]))  # 1
print(p3.feedforward([1, 0]))  # 1
print(p3.feedforward([1, 1]))  # 0
