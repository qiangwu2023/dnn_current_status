import numpy as np
import scipy.optimize as spo
def g_L(train_data,X_test,Lambda_U):
    Z_train = train_data['Z']
    X_train = train_data['X']
    De_train = train_data['De']
    def GF(*args):
        b = args[0]
        Lam1 = Lambda_U * np.exp( Z_train*b[6] + np.dot(X_train,b[0:5]) + b[5]*np.ones(X_train.shape[0]))
        loss_fun = -np.mean(De_train*np.log(1-np.exp(-Lam1)+1e-5) - (1-De_train)*Lam1) 
        return loss_fun
    linear_para = spo.minimize(GF,np.zeros(7),method='SLSQP')['x']
    print('linear_parameter=', linear_para[0:6])
    g_train = np.dot(X_train,linear_para[0:5]) + linear_para[5]*np.ones(X_train.shape[0])
    g_test = np.dot(X_test,linear_para[0:5]) + linear_para[5]*np.ones(X_test.shape[0])
    return {'linear_para': linear_para[0:6],
        'g_train': g_train,
        'g_test': g_test,
        'beta': linear_para[6]
    }
