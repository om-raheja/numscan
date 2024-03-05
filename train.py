#!/usr/bin/env python3

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

digits = load_digits()

# input
x = digits.images.reshape((len(digits.images), -1))

# what the output should be
y = digits.target

mlp = MLPClassifier(hidden_layer_sizes=(18,),
                    activation='logistic',
                    alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1,
                    verbose=True)

mlp.fit(x, y)

fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("iterations")
axes.set_ylabel("loss")
plt.show()

pickle.dump(mlp.coefs_, open( 'weights.pkl', 'wb'))
