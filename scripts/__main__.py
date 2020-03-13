from scripts.NN2 import NN
from .io import read_positives, read_negatives, balance_inputs, one_hot_encoding
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python -m scripts <positives filepath> <negatives filepath>")
    sys.exit(0)

positives = read_positives(sys.argv[1])
negatives = read_negatives(sys.argv[2])
pos_sample,neg_sample = balance_inputs(positives, negatives)

# code from https://github.com/erikdelange/Neural-networks-in-numpy/blob/master/network1.py
# my neural network code ran but did not learn : see /scripts/NN.py
network = NN((8, 3, 8))
X = np.identity(8)
Y = np.identity(8)

error = network.train(X, Y, 50000, learning_rate=0.2)

plt.plot(error)
plt.xlabel("training iterations")
plt.ylabel("mse")

Y_hat = network.forward(X)

print("predict:", Y_hat.T)
print("desired:", Y.T)
print("loss   :", (Y - Y_hat).T)

plt.show()