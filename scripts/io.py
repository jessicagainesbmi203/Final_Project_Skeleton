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



















