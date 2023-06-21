import numpy as np
from Beta_estimate import Beta_est
from C_estimation import C_est
from I_spline import I_S
from g_linear import g_L
def Est_linear(X_test,Z_train,X_train,U_train,De_train,Beta0,nodevec,m,c0):
    Lambda_U = I_S(m, c0, U_train, nodevec)
    C_index = 0
    for loop in range(500):
        print('linear_iteration time=', loop)
        g_X = g_L(X_test,Z_train,X_train,De_train,Lambda_U,Beta0)
        g_train = g_X['g_train']
        c1 = C_est(m,U_train,De_train,Z_train,Beta0,g_train,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Beta1 = Beta_est(De_train,Z_train,Lambda_U,g_train)
        print('Beta=', Beta1)
        print('c=', c1)
        if (np.max(abs(Beta0-Beta1)) <= 0.01):
            C_index = 1
            break
        c0 = c1
        Beta0 = Beta1
    return {
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'c': c1,
        'Beta': Beta1,
        'C_index': C_index,
    }





