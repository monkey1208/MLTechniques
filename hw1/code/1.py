import numpy as np
import matplotlib.pyplot as plt
import ipdb
def phi1(x):
	return 2*(x[1]**2)-4*x[0]+2
def phi2(x):
	return x[0]**2-2*x[1]-3
X = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
Z = np.zeros(X.shape)
for i, x in enumerate(X):
	Z[i, 0] = phi1(x)
	Z[i, 1] = phi2(x)
plt.scatter(Z[:3,0],Z[:3,1])
plt.scatter(Z[3:,0],Z[3:,1], cmap='g')
plt.plot([5, 5], [-8, 4], 'k-', color = 'r')
plt.ylim(-8,4)
plt.xlabel('z1')
plt.ylabel('z2')
plt.show()
ipdb.set_trace()
