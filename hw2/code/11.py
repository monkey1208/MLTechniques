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
    np.random.seed(1208)
    X, Y = readfile(sys.argv[1])
    X = addx0(X)
    X_train, Y_train, X_test, Y_test = splitData(X, Y)
    lamda = [0.05,0.5,5,50,500]
    E_in = []
    E_out = []
    Train_out = np.zeros(400)
    Test_out = np.zeros(100)
    for ld in lamda:
        for it in range(250):
            idx = np.random.randint(400, size=400)
            X_tra = X_train[idx]
            Y_tra = Y_train[idx]
            K = np.dot(X_tra.T, X_tra)
            A = (np.identity(K.shape[0])*ld+K)
            w = LA.inv(A).dot(X_tra.T).dot(Y_tra)
            train_out = np.sign(X_train.dot(w))
            test_out = np.sign(X_test.dot(w))
            
            Train_out += train_out
            Test_out += test_out
        Train_out = np.sign(Train_out)
        Test_out = np.sign(Test_out)
        E_in.append((Train_out != Y_train).sum()/Y_train.shape[0])
        E_out.append((Test_out != Y_test).sum()/Y_test.shape[0])
print(lamda)
print(E_in)
print(E_out)
