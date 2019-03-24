
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from sklearn.svm import SVC	
X = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
Y = np.array([-1,-1,-1,1,1,1,1])
N = X.shape[0]

svm = SVC(C=1e10, kernel='poly', coef0=1, degree=2, gamma=1)
svm.fit(X,Y)
alpha = np.zeros(N)
alpha[svm.support_] = Y[svm.support_]*svm.dual_coef_
sv_idx = svm.support_[0]
xs = X[sv_idx]
ys = Y[sv_idx]
coef = svm.dual_coef_[0]
b = ys - np.dot((1+np.dot(svm.support_vectors_,xs))**2, coef)# - []# ys - sum(alpha y K(xn,xs))
from sympy import *
expression = 0
x1 = Symbol('v1')
x2 = Symbol('v2')
for i, sv in enumerate(svm.support_vectors_):
	expression += coef[i]*(1+np.dot(sv,[x1,x2]))**2
expression += b
print('b = {}'.format(b))
print(expression.expand())
