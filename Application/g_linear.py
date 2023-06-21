import numpy as np
import scipy.optimize as spo
def g_L(X_test,Z_train,X_train,De_train,Lambda_U,Beta0):
    def GF(*args):
        b = args[0]
        Lam1 = Lambda_U * np.exp(Z_train[:,0]*Beta0[0] + Z_train[:,1]*Beta0[1] + np.dot(X_train, b[0:4]) + b[4]*np.ones(X_train.shape[0]))
        loss_fun = -np.mean(De_train*np.log(1-np.exp(-Lam1)+1e-5) - (1-De_train)*Lam1) 
        return loss_fun
    linear_para = spo.minimize(GF,np.zeros(5),method='SLSQP')['x']
    print('linear_parameter=', linear_para)
    g_train = np.dot(X_train,linear_para[0:4]) + linear_para[4]*np.ones(X_train.shape[0])
    g_test = np.dot(X_test,linear_para[0:4]) + linear_para[4]*np.ones(X_test.shape[0])
    return {'linear_para': linear_para,
        'g_train': g_train,
        'g_test': g_test
    }
