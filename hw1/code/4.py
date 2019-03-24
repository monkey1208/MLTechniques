
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from sklearn.svm import SVC	
def phi1(x):
	return 2*(x[1]**2)-4*x[0]+2
def phi2(x):
	return x[0]**2-2*x[1]-3
X = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
Y = np.array([-1,-1,-1,1,1,1,1])
N = X.shape[0]

svm = SVC(C=1e9, kernel='poly', coef0=1, degree=2, gamma=1)
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
	expression += coef[i]*((1+np.dot(sv,[x1,x2]))**2)
expression += b
print('b = {}'.format(b))
print(expression.expand())


def g(x):
	r = np.sqrt(2)
	return np.array([1,r*x[0], r*x[1], x[0]**2, x[0]*x[1], x[1]*x[0], x[1]**2])
def G(x1, x2, k, b):
	r = np.sqrt(2)
	return k[0]+k[1]*r*x1+k[2]*r*x2+k[3]*(x1**2)+(k[4]+k[5])*x1*x2+k[6]*(x2**2)+b
def phi(x1, x2):
	return 2*(x2**2)-4*x1-3
x4 = np.array([g(i) for i in X])
b = Y[sv_idx] - np.dot(coef, np.dot(x4[svm.support_], x4[sv_idx]))
w = np.dot(coef, x4[svm.support_])
print(w)

d = 3
x = np.linspace(-d, d, 1000)
y = np.linspace(-d, d, 1000)
xx, yy = np.meshgrid(x,y)

plt.contour(xx, yy, G(xx, yy, w, b), 0)
plt.contour(xx, yy, phi(xx, yy), 0, cmap='Reds')
plt.scatter(X[:3,0],X[:3,1])
plt.scatter(X[3:,0],X[3:,1], cmap='g')
plt.show()

