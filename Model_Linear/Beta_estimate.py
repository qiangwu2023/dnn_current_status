# Estimate Beta under known g(X), Lambda(U)
import numpy as np
import scipy.optimize as spo
# =====local optimum=====
def Beta_est(De, Z, Lambda_U, g_X):
    def BF(*args):
        Lam = Lambda_U * np.exp(Z * args[0] + g_X)
        Loss_F = np.mean(-De * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De) * Lam)
        return Loss_F
    result = spo.minimize(BF,np.zeros(1),method='SLSQP')
    return result['x']
