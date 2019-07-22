import numpy as np
from read_data import read_nolabel_data
import numpy.linalg as LA
import matplotlib.pyplot as plt
import sys
class KMeans():
    def __init__(self, k):
        self.k = k
        self.old_clusters = [-np.ones(1)]*k
        self.old_error = 0
        self.clusters = []
    def pointDistance(self, X1,X2): # X1 train, X2 test
        X1 = X1.reshape(X1.shape[0],1,X1.shape[-1])
        X2 = X2.reshape(X2.shape[0],1,X2.shape[-1])
        X1 = np.tile(X1,(1,X2.shape[0],1))
        X2 = np.tile(X2,(1,X1.shape[0],1))
        X1 = X1.transpose(1,0,-1)
        D = X2 - X1
        dist = LA.norm(D,axis=-1)**2
        return dist
    def error(self, X):
        dist = self.pointDistance(self.mu, X)
        error = dist.min(1).mean()
        return error
    def fit(self, X):
        tmp = np.arange(X.shape[0])
        np.random.shuffle(tmp)
        idx = tmp[:self.k]
        self.mu = X[idx]
        # start iteration
        while 1:
            # clustering
            dist = self.pointDistance(self.mu, X)
            cluster = dist.argsort()[:,0]
            for k in range(self.k):
                self.clusters.append((cluster == k).nonzero()[0])
            #if self.clusters[0].shape[0] == self.old_clusters[0].shape[0]:
            #    (self.clusters[0] == self.old_clusters[0]).sum() == self.clusters[0].shape[0]
            #    if flag:
            #        break
            # optimize mu
            for k in range(self.k):
                datas = X[self.clusters[k]]
                self.mu[k] = datas.mean(0)
            self.old_clusters = self.clusters.copy()
            self.clusters = []
            error = self.error(X)
            if error == self.old_error:
                break
            self.old_error = error
        # compute Error
        ein = self.error(X)
        return self.mu, ein
        
path = sys.argv[1]
X = read_nolabel_data(path)
K = [2,4,6,8,10]
Ein = []
Var = []
np.random.seed(1208)
for k in K:
    ein = []
    for it in range(500):
        kmeans = KMeans(k)
        mu, err = kmeans.fit(X)
        ein.append(err)
    var = np.std(ein)
    var = np.var(ein)
    ein = np.average(ein)
    print('k = {}, Ein = {}, Var = {}'.format(k, ein, var))
    Ein.append(ein)
    Var.append(var)
plt.plot(K, Ein)
plt.title('Ein v.s. k')
plt.xlabel('k')
plt.ylabel('Ein')
plt.xticks(range(2,11,2))
plt.savefig('15.png')
plt.clf()
plt.figure(figsize=(8,5))
plt.plot(K, Var)
plt.title('Ein variance v.s. k')
plt.xlabel('k')
plt.ylabel('Ein var')
plt.xticks(range(2,11,2))
plt.savefig('16.png')
plt.clf()
