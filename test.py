import numpy as np
import perceptron

print(perceptron.__doc__)

if __name__ == "__main__":
    p3 = perceptron.Perceptron(3, 2, 1)
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
