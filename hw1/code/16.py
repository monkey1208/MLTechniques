import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import random
import sys
def readdata(path):
	X, Y = [], []
	with open(path, 'r') as f:
		for line in f.readlines():
			line = line.strip().split()
			X.append([float(line[1]), float(line[2])])
			Y.append(int(float(line[0])))
	return np.array(X), np.array(Y)
X_train, Y_train = readdata(sys.argv[1])
#X_test, Y_test = readdata('../data/features.test')
Gamma = [-2,-1,0,1,2]
times = {-2:0, -1:0, 0:0, 1:0, 2:0}
random.seed(211)
for time in range(100):
	idx = random.sample(range(X_train.shape[0]), 1000)
	Xval, Yval = X_train[idx], Y_train[idx]
	Xtrain, Ytrain = np.delete(X_train, idx, axis=0), np.delete(Y_train, idx, axis=0)
	Eval, ga = 1, -10
	for gamma in Gamma:
		svm = SVC(C=0.1, kernel='rbf', coef0=0, gamma=10**gamma)
		svm.fit(Xtrain, Ytrain==0)
		err = np.sum(svm.predict(Xval) != (Yval == 0))/Yval.shape[0]
		if err < Eval:
			Eval = err
			ga = gamma
	times[ga] += 1
	
plt.bar(Gamma, list(times.values()))
plt.xlabel('gamma')
plt.ylabel('times')
plt.title('times v.s. gamma')
plt.savefig('16.png')
