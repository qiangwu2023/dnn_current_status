import numpy as np
import scipy.optimize as spo
from B_spline3 import B_S

def g_A(X_test,Z_train,X_train,De_train,Lambda_U,Beta0,m0,nodevec0):
    B_0 = B_S(m0, X_train[:,0], nodevec0)
    B_1 = B_S(m0, X_train[:,1], nodevec0)
    B_2 = B_S(m0, X_train[:,2], nodevec0)
    B_3 = B_S(m0, X_train[:,3], nodevec0)
    def GA(*args):
        b = args[0]
        Lam1 = Lambda_U * np.exp( Z_train[:,0]*Beta0[0] + Z_train[:,1]*Beta0[1] + np.dot(B_0, b[0:(m0+4)]) + np.dot(B_1, b[(m0+4):(2*(m0+4))]) + np.dot(B_2, b[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_3, b[(3*(m0+4)):(4*(m0+4))]) + b[4*(m0+4)]*np.ones(X_train.shape[0]))
        loss_fun = -np.mean(De_train*np.log(1-np.exp(-Lam1)+4e-5) - (1-De_train)*Lam1) 
        return loss_fun
    param = spo.minimize(GA,np.zeros(4*(m0+4)+1),method='SLSQP')['x']
    g_train = np.dot(B_0, param[0:(m0+4)]) + np.dot(B_1, param[(m0+4):(2*(m0+4))]) + np.dot(B_2, param[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_3, param[(3*(m0+4)):(4*(m0+4))]) + param[4*(m0+4)]*np.ones(X_train.shape[0])
    g_test = np.dot(B_S(m0, X_test[:,0], nodevec0), param[0:(m0+4)]) + np.dot(B_S(m0, X_test[:,1], nodevec0), param[(m0+4):(2*(m0+4))]) + np.dot(B_S(m0, X_test[:,2], nodevec0), param[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_S(m0, X_test[:,3], nodevec0), param[(3*(m0+4)):(4*(m0+4))]) + param[4*(m0+4)]*np.ones(X_test.shape[0])
    return{
        'g_train': g_train,
        'g_test': g_test
    }