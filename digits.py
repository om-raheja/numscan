#!/usr/bin/env python3

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

digits = load_digits()

# input
x = digits.images.reshape((len(digits.images), -1))

# what the output should be
y = digits.target

x_train = x[:1500]
y_train = y[:1500]

x_test = x[1500:]
y_test = y[1500:]

mlp = MLPClassifier(hidden_layer_sizes=(15,),
                    activation='logistic',
                    alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1,
                    verbose=True)

mlp.fit(x_train, y_train)

fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("iterations")
axes.set_ylabel("loss")
plt.show()

