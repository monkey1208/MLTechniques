import numpy as np

def readfile(path):
    X, Y = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            y = int(line[-1])
            x = np.array(line[:-1]).astype(float)
            Y += [y]
            X += [x]
    return np.array(X), np.array(Y)
if __name__ == '__main__':
    import sys
    X, Y = readfile(sys.argv[1])
