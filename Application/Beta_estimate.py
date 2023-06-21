import numpy as np
import scipy.optimize as spo
def Beta_est(De, Z, Lambda_U, g_X):
    def BF(*args):
        a = args[0]
        Lam = Lambda_U * np.exp(np.dot(Z, a)+ g_X)
        Loss_F = np.mean(-De * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De) * Lam)
        return Loss_F
    result = spo.minimize(BF,np.zeros(2),method='SLSQP')['x']
    return result


