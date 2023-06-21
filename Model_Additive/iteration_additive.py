import numpy as np
from Beta_estimate import Beta_est
from C_estimation import C_est
from I_spline import I_S
from g_additive import g_A

def Est_additive(train_data,X_test,Beta0,nodevec,m,c0,m0,nodevec0):
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    Beta0 = np.array([Beta0])
    # Given an initial value c0, calculate an initial value of Lambda(U)
    Lambda_U = I_S(m, c0, U_train, nodevec)
    C_index = 0
    for loop in range(100):
        print('additive_iteration time=', loop)
        # The initial values of Beta and Lambda(U) are given, g(X) needs to be estimated
        g_X = g_A(train_data,X_test,Lambda_U,Beta0,m0,nodevec0)
        g_train = g_X['g_train']
        # Estimate Lambda(U)
        c1 = C_est(m,U_train,De_train,Z_train,Beta0,g_train,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        # Estimate Beta
        Beta1 = Beta_est(De_train,Z_train,Lambda_U,g_train)
        print('Beta=', Beta1)
        print('c=', c1)
        # Convergence condition
        if (abs(Beta0-Beta1) <= 0.001):
            C_index = 1
            break
        c0 = c1
        Beta0 = Beta1
    return{
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'c': c1,
        'Beta': Beta1,
        'C_index': C_index,
    }





