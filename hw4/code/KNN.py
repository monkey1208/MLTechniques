import numpy as np
import numpy.linalg as LA
from read_data import read_data
import matplotlib.pyplot as plt
import sys
path = sys.argv[1]
test_path = sys.argv[2]
X, Y = read_data(path)
X_test, Y_test = read_data(test_path)
def pointDistance(X1,X2): # X1 train, X2 test
    X1 = X1.reshape(X1.shape[0],1,X1.shape[-1])
    X2 = X2.reshape(X2.shape[0],1,X2.shape[-1])
    X1 = np.tile(X1,(1,X2.shape[0],1))
    X2 = np.tile(X2,(1,X1.shape[0],1))
    X1 = X1.transpose(1,0,-1)
    D = X2 - X1
    dist = LA.norm(D,axis=-1)**2
    return dist

##############
# problem 11 #
# KNN Ein    #
##############
print('***** K-NN *****')
dist = pointDistance(X,X) # Ein
dist = dist.argsort()
K = [1,3,5,7,9]
Ein = []
for k in K:
    neighbors = dist[:,:k]
    predict = Y[neighbors]
    predict = predict.sum(1)
    predict = np.sign(predict)
    error = (predict != Y).mean()
    Ein.append(error)
    print('k = {} : Ein = {}'.format(k, error))
plt.plot(K, Ein)
plt.title('Ein v.s. k')
plt.xticks(range(1,10,2))
plt.xlabel('k')
plt.ylabel('Ein')
plt.savefig('11.png')
plt.clf()
##############
# problem 12 #
# KNN Eout   #
##############
dist = pointDistance(X,X_test) # Eout
dist = dist.argsort()
Eout = []
for k in K:
    neighbors = dist[:,:k]
    predict = Y[neighbors]
    predict = predict.sum(1)
    predict = np.sign(predict)
    error = (predict != Y_test).mean()
    Eout.append(error)
    print('k = {} : Eout = {}'.format(k, error))
plt.plot(K, Eout)
plt.title('Eout v.s. k')
plt.xticks(range(1,10,2))
plt.xlabel('k')
plt.ylabel('Eout')
plt.savefig('12.png')
plt.clf()

Gamma = [.001, 0.1, 1, 10, 100]
##############
# problem 13 #
# KNN Ein    #
##############
print('***** Uniform NN *****')
dist = pointDistance(X,X) # Ein
Ein = []
for gamma in Gamma:
    expdist = np.exp(-gamma*dist)
    predict = np.sign(np.dot(expdist, Y))
    error = (predict != Y).mean()
    Ein.append(error)
    print('gamma = {} : Ein = {}'.format(gamma, error))

plt.plot(Gamma, Ein)
plt.title('Ein v.s. gamma')
plt.xscale('log')
plt.xlabel('gamma')
plt.ylabel('Ein')
plt.savefig('13.png')
plt.clf()
##############
# problem 14 #
# KNN Eout   #
##############
dist = pointDistance(X,X_test) # Eout
Eout = []
for gamma in Gamma:
    expdist = np.exp(-gamma*dist)
    predict = np.sign(np.dot(expdist, Y))
    error = (predict != Y_test).mean()
    Eout.append(error)
    print('gamma = {} : Eout = {}'.format(gamma, error))

plt.plot(Gamma, Eout)
plt.title('Eout v.s. gamma')
plt.xscale('log')
plt.xlabel('gamma')
plt.ylabel('Eout')
plt.savefig('14.png')
plt.clf()
