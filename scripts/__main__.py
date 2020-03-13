from scripts.NN2 import NN
from .io import encode_data
from .k_fold import k_fold_learning_rate, get_k_split_data
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python -m scripts <positives filepath> <negatives filepath>")
    sys.exit(0)

def demonstrate_autoencoder():
    """
    Demonstrate that neural network code can recreate a 8 x 8 identity matrix after compressing down to 3 nodes
    """
    # code from https://github.com/erikdelange/Neural-networks-in-numpy/blob/master/network1.py
    # my neural network code ran but did not learn : see /scripts/NN.py
    network = NN((8,3,8))
    # autoencoder input
    X = np.identity(8)
    # autoencoder output
    Y = np.identity(8)
    # train the model
    print(network.weight)
    error = network.train(X, Y, 50000, learning_rate=0.2)

    plt.plot(error)
    plt.xlabel("training iterations")
    plt.ylabel("mse")

    # use model to predict
    Y_hat = network.forward(X)
    print("predict:", Y_hat.T)
    print("desired:", Y.T)
    print("loss   :", (Y - Y_hat).T)

    plt.show()
    
def DNA_prediction():
    """
    Preprocess data, encode sequences using one-hot encoding, and 
        train a neural network to predict whether the sequence is
        a true binding site
    """
    positives = read_positives(sys.argv[1])
    negatives = read_negatives(sys.argv[2])
    pos_sample,neg_sample = balance_inputs(positives, negatives)
    one_hot_pos = list()
    for sample in pos_sample:
        one_hot_pos.append(one_hot_encoding(sample))
    flat_pos = np.asarray(flatten_inputs(one_hot_pos))
    one_hot_neg = list()
    for sample in neg_sample:
        one_hot_neg.append(one_hot_encoding(sample))
    flat_neg = np.asarray(flatten_inputs(one_hot_neg))
    pos_length = flat_pos.shape[0]
    neg_length = flat_neg.shape[0]
    pos_output = np.full((pos_length,1),fill_value=1)
    neg_output = np.zeros((neg_length,1))
    inputs = np.concatenate((flat_pos,flat_neg),axis=0)
    outputs = np.concatenate((pos_output,neg_output))
    n_features = inputs.shape[1]
    neural_net = NN((n_features,40,1))
    loss =  neural_net.train(inputs,outputs,50000,learning_rate=0.2)
    y_hat = neural_net.forward(inputs)
    print("predict:", y_hat.T)
    print('training loss')
    print(loss[-1])

#demonstrate_autoencoder()
#DNA_prediction()
encoded_pos, encoded_neg = encode_data(sys.argv[1],sys.argv[2])
sets = get_k_split_data(10, encoded_pos, encoded_neg)
k_fold_learning_rate(10,sets)
