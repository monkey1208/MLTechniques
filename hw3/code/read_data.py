import numpy as np
def read_data(path):
    with open(path, 'r') as f:
        X, Y = [], []
        for line in f.readlines():
            line = line.strip().split()
            X.append(line[:-1])
            Y.append(line[-1])
    return np.array(X).astype(float), np.array(Y).astype(int)
