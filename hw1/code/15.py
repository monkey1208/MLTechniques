import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC	
from sklearn.gaussian_process.kernels import RBF
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
C = [-2,-1,0,1,2]
dis = []
rbf = RBF(length_scale=(np.sqrt(1/160)))
for c in C:
	svm = SVC(C=10**c, kernel='rbf', coef0=0, gamma=80)
	svm.fit(X_train, Y_train==0)
	sv = svm.support_vectors_
	alpha = svm.dual_coef_[0]
	K = rbf(sv)
	W2 = np.dot(np.dot(alpha, K), alpha)
	dis.append(1/(W2**0.5))

plt.plot(C, dis, '-o')
plt.xlabel('C')
plt.ylabel('distance')
plt.title('distance v.s. log10C')
plt.savefig('15.png')
