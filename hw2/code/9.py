from readfile import readfile
import numpy as np
import numpy.linalg as LA
import sys
def splitData(X, Y, cnt=400):
    X_train, Y_train = X[:400], Y[:400]
    X_test, Y_test = X[400:], Y[400:]
    return X_train, Y_train, X_test, Y_test
def addx0(X):
    return np.insert(X, 0, 1, axis=1)

if __name__ == '__main__':
    import ipdb
    X, Y = readfile(sys.argv[1])
    X = addx0(X)
    X_train, Y_train, X_test, Y_test = splitData(X, Y)
    #K = np.dot(X_train, X_train.T)
    K = np.dot(X_train.T, X_train)
    lamda = [0.05,0.5,5,50,500]
    E_in = []
    E_out = []
    for ld in lamda:
        #A = (np.identity(400)*ld+K)
        A = (np.identity(K.shape[0])*ld+K)
        #beta = LA.inv(A).dot(Y_train)
        w = LA.inv(A).dot(X_train.T).dot(Y_train)
        #w = beta.dot(X_train)
        train_out = np.sign(X_train.dot(w))
        test_out = np.sign(X_test.dot(w))
        E_in.append((train_out != Y_train).sum()/Y_train.shape[0])
        E_out.append((test_out != Y_test).sum()/Y_test.shape[0])
print(lamda)
print(E_in)
print(E_out)
