import numpy as np
import os
def format_output(y_pred):
    """
    Takes as input a np.ndarray of predictions, which may be real numbers, and rounds them to the nearest integer between 1 and 5
    """
    length = y_pred.shape[0]
    fives = np.full(length, 5)
    ones = np.full(length, 1)
    return np.maximum(ones, np.minimum(fives, np.rint(y_pred)))

def write_output(y_pred, fname):
    fname = 'output/' + fname
    if os.getcwd()[-3:] == 'src':
        fname = '../' + fname
    with open(f'{fname}.csv', 'w') as f:
        f.write("index,stars\n")
        i = 0
        for val in y_pred:
            f.write(f'{i},{val}\n')
            i += 1
