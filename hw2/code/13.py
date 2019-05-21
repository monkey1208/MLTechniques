from readfile import readfile
import numpy as np
import sys
import matplotlib.pyplot as plt
def predict(X, Alpha, G):
    t = len(Alpha)
    thetas = [i['theta'] for i in G]
    S = [i['s'] for i in G]
    dim = [i['d'] for i in G]
    data = X[:,dim]
    Y = (S*np.sign(data - thetas))
    Y = Y.dot(Alpha)
    return np.sign(Y)

if __name__ == '__main__':
    X_train, Y_train = readfile(sys.argv[1])
    X_test, Y_test = readfile(sys.argv[2])
    iterations = 300
    X_train_d0 = np.array(sorted(X_train[:,0]))
    X_train_d1 = np.array(sorted(X_train[:,1]))
    theta_d0 = (X_train_d0[:-1]+X_train_d0[1:])/2
    theta_d0 = np.concatenate(([X_train_d0[0]-1], theta_d0))
    theta_d1 = (X_train_d1[:-1]+X_train_d1[1:])/2
    theta_d1 = np.concatenate(([X_train_d1[0]-1], theta_d1))
    thetas = np.array([theta_d0, theta_d1])
    U = np.ones(X_train.shape[0])/X_train.shape[0]
    label = np.tile(Y_train, (Y_train.shape[0],1))
    Ein = []
    Eout = []
    Ein_u = []
    Alpha = []
    G = []
    all_U = []
    for it in range(iterations):
        # find the best gt
        error = 1
        S = 0
        best_theta = 1208
        best_dim = 1208
        for dim in range(thetas.shape[0]):
            Thet = thetas[dim]
            data = X_train[:,dim]
            data = np.tile(data, (100,1))
            Theta = np.tile(Thet,(100,1)).T 
            u_weight = np.tile(U, (100,1))
            output = np.sign(data-Theta)
            for s in [-1,1]:
                directional_output = output*s
                err = np.sum((label != directional_output)*u_weight, axis=1)/label.shape[1]
                idx = err.argmin()
                min_err = err[idx]
                if min_err < error:
                    error = min_err
                    S = s
                    best_theta = Thet[idx]
                    best_dim = dim
        incorrect_data = (S*np.sign(X_train[:,best_dim]-best_theta) != Y_train)
        eps = U.dot(incorrect_data)/U.sum()
        factor = ((1-eps)/eps)**0.5
        alpha = np.log(factor)
        U[ incorrect_data ] *= factor
        U[ ~incorrect_data ] /= factor
        all_U.append(U.sum())
        Ein_u.append(error)
        Alpha.append(alpha)
        G.append({'s':S, 'd':best_dim, 'theta':best_theta})
        Y_predict = predict(X_train, Alpha, G)
        Ein.append((Y_predict !=Y_train).mean())
        Y_predict = predict(X_test, Alpha, G)
        Eout.append((Y_predict !=Y_test).mean())
    plt.plot(range(1,iterations+1), Ein_u)
    plt.xlabel('t')
    plt.ylabel('Ein_u')
    plt.savefig('13.png')
    plt.clf()
    plt.plot(range(1,iterations+1), Ein)
    plt.xlabel('t')
    plt.ylabel('Ein')
    plt.savefig('14.png')
    plt.clf()
    plt.plot(range(1,iterations+1), Eout)
    plt.xlabel('t')
    plt.ylabel('Eout')
    plt.savefig('16.png')
    plt.clf()
    plt.plot(range(1,iterations+1), all_U)
    plt.xlabel('t')
    plt.ylabel('U')
    plt.savefig('15.png')
    print('Ein_u = {}'.format(Ein_u[-1]))
    print('Ein = {}'.format(Ein[-1]))
    print('Eout = {}'.format(Eout[-1]))
    print('U = {}'.format(all_U[-1]))
