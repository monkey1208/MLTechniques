
def read_data(path):
    import numpy as np
    X = []
    Y = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            X.append(data[:-1])
            Y.append(data[-1])
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(int)
    return X, Y
def read_nolabel_data(path):
    import numpy as np
    X = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            X.append(data)
    X = np.array(X).astype(float)
    return X
