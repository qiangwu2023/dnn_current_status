import numpy as np
import scipy.optimize as spo
from I_spline import I_U
def C_est(m, U, De, Z, Beta, g_X, nodevec):
    Iu = I_U(m, U, nodevec)
    def LF(*args):
        a = args[0]
        Lam1 = np.dot(Iu,a) * np.exp(Z[:,0] * Beta[0] + Z[:,1] * Beta[1] + g_X)
        Loss_F1 = np.mean(-De * np.log(1 - np.exp(-Lam1) + 1e-5) + (1-De)*Lam1)
        return Loss_F1
    bnds = []
    for i in range(m+3):
        bnds.append((0,1000))
    result = spo.minimize(LF,np.ones(m+3),method='SLSQP',bounds=bnds)
    return result['x']


