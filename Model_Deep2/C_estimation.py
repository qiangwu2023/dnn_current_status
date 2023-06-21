# Under the known g(X), estimate the control points c_1,c_2,...,c_(m-p) of the integrated splines
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
# The unknown coefficients here are all required to be greater than or equal to 0, which is a constraint
def C_est(m, U, De, Z, Beta, g_X, nodevec):
    Iu = I_U(m, U, nodevec)
    def LF(*args):
        a = args[0]
        Lam1 = np.dot(Iu,a) * np.exp(Z*Beta+g_X)
        Loss_F1 = np.mean(-De * np.log(1-np.exp(-Lam1)+1e-5) + (1-De)*Lam1)
        return Loss_F1
    bnds = []
    for i in range(m+3):
        bnds.append((0,1000))
    result = spo.minimize(LF,np.ones(m+3),method='SLSQP',bounds=bnds)
    return result['x']


