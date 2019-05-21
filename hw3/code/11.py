from read_data import read_data
import sys
import numpy as np
import matplotlib.pyplot as plt

class TreeNode():
    def __init__(self, theta, dim, depth, label=None):
        self.theta = theta
        self.dim = dim
        self.depth = depth
        self.label = label
        self.left = None
        self.right = None
    def printNode(self):
        offset = ' '*10*self.depth
        print(offset+'*'*20)
        if self.label == None:
            print(offset+'Branch here, depth = {}'.format(self.depth))
            print(offset+'theta = {}'.format(self.theta))
            print(offset+'dim = {}'.format(self.dim))
        else:
            print(offset+'Leaf Node, depth = {}'.format(self.depth))
            print(offset+'label value = {}'.format(self.label))
        print(offset+'*'*20)
class DecisionTree():
    def __init__(self, max_height=np.inf):
        self.root = None
        self.max_height = max_height
        self.height = 0
    def calculate_theta(self, X):
        theta0 = np.sort(X[:,0])
        theta1 = np.sort(X[:,1])
        theta0 = (theta0[1:]+theta0[:-1])/2
        theta0 = np.append(theta0[0]-1, theta0)
        theta0 = np.append(theta0, theta0[-1]+1)
        theta1 = (theta1[1:]+theta1[:-1])/2
        theta1 = np.append(theta1[0]-1, theta1)
        theta1 = np.append(theta1, theta1[-1]+1)
        thetas = np.array([theta0, theta1])
        return thetas
        
    def gini_index(self, Y):
        N = Y.shape[0]
        pos = (Y > 0).sum()
        neg = (Y < 0).sum()
        if N == 0:
            return 1
        return 1-((pos/N)**2)-((neg/N)**2)
    def terminate(self, X, Y, depth):
        # when every data has the same label
        if depth+1 >= self.max_height:
            return True
        if len(Y) == np.abs(Y.sum()):
            return True
        elif (X != X[0]).sum() == 0:
            return True
        return False
    def decision_stump(self, X, Y):
        best_err = np.inf
        best_theta = best_dim = 0
        X_left = Y_left = X_right = Y_right = None
        Thetas = self.calculate_theta(X)
        for dimension in range(X.shape[1]):
            thetas = Thetas[dimension]
            for theta in thetas:
                YL = np.array(Y[X[:,dimension] < theta])
                YR = np.array(Y[X[:,dimension] >= theta])
                impurityL = self.gini_index(YL)
                impurityR = self.gini_index(YR)
                criteria = YL.shape[0]*impurityL + YR.shape[0]*impurityR
                if criteria < best_err:
                    best_err = criteria
                    best_theta = theta
                    best_dim = dimension
                    Y_left = YL
                    Y_right = YR
                    X_left = X[X[:,dimension] < theta]
                    X_right = X[X[:,dimension] >= theta]
        return X_left, Y_left, X_right, Y_right, best_theta, best_dim, best_err
    def decision_tree(self, X, Y, depth=0):
        if self.terminate(X, Y, depth):
            pos = (Y >= 0)
            neg = (Y < 0)
            label = Y[pos][0] if pos.sum()>neg.sum() else Y[neg][0]
            return TreeNode(None, None, depth, label)
        XL, YL, XR, YR, theta, dim, err = self.decision_stump(X, Y)
        node = TreeNode(theta, dim, depth)
        node.left = self.decision_tree(XL, YL, depth+1)
        node.right = self.decision_tree(XR, YR, depth+1)
        return node
    def fit(self, X, Y):
        self.root = self.decision_tree(X, Y)
        self.find_height(self.root)
        return self.root
    def find_height(self, node):
        if node.label != None:
            if node.depth+1 > self.height:
                self.height = node.depth+1
            return
        self.find_height(node.left)
        self.find_height(node.right)
    def _predict(self, node, X):
        if node.label != None:
            return node.label
        if X[node.dim] >= node.theta:
            return self._predict(node.right, X)
        else:
            return self._predict(node.left, X)
    def predict(self, X):
        if self.root == None:
            print('DTree not fit.')
            return
        Y_pred = []
        for x in X:
            Y_pred.append(self._predict(self.root, x))
        return np.array(Y_pred)
    def printTree(self, node):
        if node == None: return
        if node.left == None and node.right == None:
            node.printNode()
            return
        node.printNode()
        self.printTree(node.left)
        self.printTree(node.right)
        

X, Y = read_data(sys.argv[1])
X_test, Y_test = read_data(sys.argv[2])
Dtree = DecisionTree()
Dtree.fit(X, Y)
Y_pred = Dtree.predict(X)
print('Ein = {}'.format((Y_pred != Y).mean()))
Y_pred = Dtree.predict(X_test)
print('Eout = {}'.format((Y_pred != Y_test).mean()))
Dtree.printTree(Dtree.root)

#############################
# Decision Tree with Height #
#############################
H = []
Ein = []
Eout = []
for height in range(Dtree.height-1, 0, -1):
    H.append(height)
    Dtree = DecisionTree(height)
    Dtree.fit(X, Y)
    Y_pred = Dtree.predict(X)
    print('***** Height = {}'.format(height))
    Ein.append((Y_pred != Y).mean())
    print('Ein = {}'.format((Y_pred != Y).mean()))
    Y_pred = Dtree.predict(X_test)
    Eout.append((Y_pred != Y_test).mean())
    print('Eout = {}'.format((Y_pred != Y_test).mean()))
plt.subplot(2,2,1)
plt.plot(H, Ein)
plt.title('height v.s Ein')
plt.xlabel('height')
plt.ylabel('Ein')
plt.ylim(0,0.3)
plt.subplot(2,2,2)
plt.plot(H, Eout)
plt.title('height v.s Eout')
plt.xlabel('height')
plt.ylabel('Eout')
plt.ylim(0,0.3)
plt.subplots_adjust(wspace =0.3, hspace =0)
plt.savefig('13_2.png')
plt.clf()
l1, = plt.plot(H, Ein, label='Ein')
l2, = plt.plot(H, Eout, label='Eout')
plt.title('height v.s. Error')
plt.xlabel('height')
plt.ylabel('Error')
plt.legend(handles=[l1, l2])
plt.xticks(np.arange(1, max(H)+1, 1))
plt.savefig('13.png')
#################
# Random Forest #
#################

Trees = 30000
np.random.seed(1208)
random_forest = []

for it in range(Trees):
    idx = np.random.randint(0, X.shape[0], int(X.shape[0]*0.8))
    X_sample, Y_sample = X[idx], Y[idx]
    Dtree = DecisionTree()
    Dtree.fit(X_sample, Y_sample)
    random_forest.append(Dtree)
Ein_g = []
Y_predict = []
Y_test_predict = []
for dtree in random_forest:
    Y_pred = dtree.predict(X)
    Y_test_pred = dtree.predict(X_test)
    Y_predict.append(Y_pred)
    Y_test_predict.append(Y_test_pred)
    Ein_g.append( (Y_pred != Y).mean() )
Y_predict = np.array(Y_predict)
Y_test_predict = np.array(Y_test_predict)
plt.clf()
plt.hist(Ein_g, bins=np.linspace(0, 0.18, 19))
plt.title('Ein(gt)')
plt.xlabel('Ein')
plt.ylabel('count')
plt.savefig('14.png')
plt.clf()
Ein_Gt = []
Eout_Gt = []
Y_pred = Y_test_pred = 0
for t in range(len(Y_predict)):
    #Y_pred = np.sign(Y_predict[:t+1].sum(0))
    Y_pred += Y_predict[t]
    Ein_Gt.append((np.sign(Y_pred) != Y).mean()  )
    #Y_pred = np.sign(Y_test_predict[:t+1].sum(0))
    Y_test_pred += Y_test_predict[t]
    Eout_Gt.append((np.sign(Y_test_pred) != Y_test).mean()  )
plt.plot(range(len(Y_predict)), Ein_Gt)
plt.title('t v.s. Ein(Gt)')
plt.ylim(0,0.25)
plt.xlabel('t')
plt.ylabel('Ein')
plt.savefig('15.png')
plt.clf()
plt.plot(range(len(Y_predict)), Eout_Gt)
plt.title('t v.s. Eout(Gt)')
plt.ylim(0,0.25)
plt.xlabel('t')
plt.ylabel('Eout')
plt.savefig('16.png')
plt.clf()
