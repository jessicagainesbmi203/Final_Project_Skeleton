from scripts.NN2 import NN
import numpy as np

def test_encoder():
    network = NN((8,3,8))
    X = np.identity(8)
    Y = np.identity(8)
    network.train(X, Y, 15000, learning_rate=0.2)
    Y_hat = network.forward(X)
    assert np.round(Y_hat) == Y