
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from sklearn.svm import SVC	
X = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
Y = np.array([-1,-1,-1,1,1,1,1])
N = X.shape[0]

svm = SVC(C=1e10, kernel='poly', coef0=1, degree=2, gamma=1)
svm.fit(X,Y)
ipdb.set_trace()

'''
Q = np.dot(Y.reshape(-1,1), Y.reshape(1,-1))
Q1 = np.dot(X, X.T)
Q = Q*Q1
P = -np.ones(N)
ipdb.set_trace()
'''
