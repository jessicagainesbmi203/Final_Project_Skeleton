import os
import numpy as np

def read_positives(filepath):
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)
    if name[1] != ".txt":
        raise IOError("%s is not a txt file"%filepath)
    # open txt file
    positives = list()
    with open(filepath, "r") as f:
        # iterate over each line in the file
        for line in f:
            trimmed_line = line.strip('\n')
            positives.append(trimmed_line)     
    return positives
    
def read_negatives(filepath):
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)
    
    if name[1] != '.fa':
        raise IOError("%s is not a .fa file"%filepath)
    # open txt file
    negatives = list()
    element = ''
    with open(filepath, "r") as f:
        # iterate over each line in the file
        for line in f:
            trimmed_line = line.strip('\n')
            if trimmed_line[0] == '>':
                if element:
                    negatives.append(element)
                element = ''
            else:
                element = element + trimmed_line
    return negatives

def balance_inputs(positive_list, negative_list):
    positives = list()
    for i in range(5):
        positives.extend(positive_list)
    length = len(positives)
    negatives = list()
    samples = np.random.choice(negative_list,length,replace=True)
    for sample in samples:
        start_base = np.random.randint(0,len(sample)-len(positives[0]))
        element = sample[start_base:start_base+len(positives[0])]
        negatives.append(element)
    return positives,negatives

def one_hot_encoding(sequence):
    encoded = np.zeros(shape=(4,len(sequence)))
    i = 0
    for base in sequence:
        if base == 'A':
            encoded[0,i] = 1
        if base == 'C':
            encoded[1,i] = 1
        if base == 'G':
            encoded[2,i] = 1
        if base == 'T':
            encoded[3,i] = 1
        i += 1
    return encoded

def flatten_inputs(inputs):
    flattened_list = list()
    for array in inputs:
        a = array.flatten()
        flattened_list.append(a)
    return flattened_list

def encode_data(filepath_pos,filepath_neg):
    """
    Read in sequences with positive and negative labels and use one-hot
        encoding to quanitfy them in preparation for neural network learning_rate
    Inputs: filepath_pos : filepath containing sequences with positive labels
        filepath_neg : filepath containing sequences with negative labels
    Output: flat_pos: a list of 1d vectors containgin all features of the observations with positive labels
        flat_neg: a list of 1d vectors containing all features of the observations with negative labels
    """
    # read in positives and negatives
    positives = read_positives(filepath_pos)
    negatives = read_negatives(filepath_neg)
    # balance positives and negatives to the same length
    pos_sample,neg_sample = balance_inputs(positives, negatives)
    # one-hot encoding
    one_hot_pos = list()
    for sample in pos_sample:
        one_hot_pos.append(one_hot_encoding(sample))
    flat_pos = flatten_inputs(one_hot_pos)
    one_hot_neg = list()
    for sample in neg_sample:
        one_hot_neg.append(one_hot_encoding(sample))
    # flatten one-hot encoded samples into 1d arrays
    flat_neg = flatten_inputs(one_hot_neg)
    return flat_pos,flat_neg

def write_results(filepath, sequences, predictions):
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)
    if name[1] != ".txt":
        raise IOError("%s is not a txt file"%filepath)
    # open txt file
    with open(filepath, "w") as f:
        for (seq, pred) in zip(sequences,predictions):
            f.write(seq + "\t" + str(pred[0]) + '\n')














