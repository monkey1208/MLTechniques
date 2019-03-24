import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC	
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
C = [-5,-3,-1,1,3]
W = []
for c in C:
	svm = SVC(C=10**c, kernel='linear', coef0=0)
	svm.fit(X_train, Y_train==2)
	w = svm.coef_[0]
	W.append(np.dot(w,w)**0.5)
plt.plot(C, W, '-o')
plt.xlabel('C')
plt.ylabel('||w||')
plt.savefig('13.png')
