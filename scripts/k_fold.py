import numpy as np
from scripts.NN2 import NN

def get_k_split_data(k, positives, negatives):
    """
    Prepare a set of positive observations and negative observations for k fold cross validation
        by splitting them into k chunks and holding one out for testing
    Inputs: k : the desired number of training and testing sets
        positives: a set of encoded obervations with positive labels
        negatives: a set of encoded observations with negative labels
    Outputs: sets: a list of k sets of training data and test data
    """
    full_length = len(positives)
    # shuffle into random order before splitting
    shuffled_pos = np.random.permutation(positives)
    shuffled_neg = np.random.permutation(negatives)
    # split into k groups
    split_length = int(np.floor(full_length / k))
    pos_chunks = list()
    neg_chunks = list()
    for i in range(k):
        chunk = shuffled_pos[i*split_length:(i+1)*split_length]
        pos_chunks.append(chunk)
        chunk = shuffled_neg[i*split_length:(i+1)*split_length]
        neg_chunks.append(chunk)
    # hold out one chunk at a time
    sets = list()
    for i in range(k):
        # compile test set
        copy_pos = pos_chunks.copy()
        hold_out_pos = copy_pos.pop(i)
        copy_neg = neg_chunks.copy()
        hold_out_neg = copy_neg.pop(i)
        pos_output = np.full((len(hold_out_pos),1),fill_value=1)
        neg_output = np.zeros((len(hold_out_neg),1))
        inputs = np.concatenate((np.asarray(hold_out_pos),np.asarray(hold_out_neg)),axis=0)
        outputs = np.concatenate((pos_output,neg_output))
        test_set = (inputs,outputs)
        # re-concat training sets
        training_set_pos = list()
        for chunk in copy_pos:
            training_set_pos.extend(chunk)
        training_set_neg = list()
        for chunk in copy_neg:
            training_set_neg.extend(chunk)
        # create outputs
        pos_output = np.full((len(training_set_pos),1),fill_value=1)
        neg_output = np.zeros((len(training_set_neg),1))
        # compile training set
        inputs = np.concatenate((np.asarray(training_set_pos),np.asarray(training_set_neg)),axis=0)
        outputs = np.concatenate((pos_output,neg_output))
        training_set = (inputs,outputs)
        sets.append((training_set,test_set))
    return sets

def k_fold_learning_rate(k,sets):
    """
    Use k fold cross validation to choose a value for learning rate
    Inputs: k : the number of data splits to test
        filepath_pos : filepath containing sequences with positive labels
        filepath_neg : filepath containing sequences with negative labels
    Outputs: learning_rates : list of all learning rates tested
        training_error: a list of the average error based on training data for each learning rate 
        testing_error: a list of the average error based on the test data for each learning rate
    """
    learning_rates = [0.05,0.10,0.15,0.20,0.25]
    training_error_list = list()
    testing_error_list = list()
    for lr in learning_rates:
        testing_error_sum = 0
        training_error_sum = 0
        for split in sets:
            training_data = split[0]
            test_data = split[1]
            n_features = training_data[0].shape[1]
            neural_net = NN((n_features,40,1))
            inputs = training_data[0]
            outputs = training_data[1]
            # train model on training data
            losses = neural_net.train(inputs,outputs,50000,learning_rate=lr)
            training_error_sum += losses[-1]
            # use model to predict on test data
            inputs = test_data[0]
            outputs = test_data[1]
            y_hat = neural_net.forward(inputs)
            # calculate error for trained model on test data
            testing_error = np.sum((y_hat - outputs) ** 2)
            testing_error_sum += testing_error
        training_error_avg = training_error_sum / k
        training_error_list.append(training_error_avg)
        testing_error_avg = testing_error_sum / k
        testing_error_list.append(testing_error_avg)
    return (learning_rates,training_error_list,testing_error_list)