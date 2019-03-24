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
Q = 2
E_in = []
for c in C:
	svm = SVC(C=10**c, kernel='poly', coef0=1, degree=Q, gamma=1)
	svm.fit(X_train, Y_train==4)
	Y = svm.predict(X_train)
	Ein = sum((Y_train==4) != Y)/len(Y_train)
	sv = svm.support_vectors_
	w = svm.dual_coef_[0]
	E_in.append(Ein)

plt.plot(C, E_in, '-o')
plt.xlabel('C')
plt.ylabel('E_in')
plt.title('Ein v.s. log10C')
plt.savefig('14.png')
