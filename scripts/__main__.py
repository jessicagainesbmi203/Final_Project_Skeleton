from scripts.NN import NeuralNetwork
from .io import read_positives, read_negatives, balance_inputs, one_hot_encoding
import sys
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python -m scripts <positives filepath> <negatives filepath>")
    sys.exit(0)

positives = read_positives(sys.argv[1])
negatives = read_negatives(sys.argv[2])
pos_sample,neg_sample = balance_inputs(positives, negatives)

# test 8x3x8 encoder
test_input = np.zeros((8,8))
for i in range(8):
    test_input[i,i] = 1
    

neural_net = NeuralNetwork(inputs=test_input,outputs=test_input,iter=10)
neural_net.fit()
neural_net.predict()
