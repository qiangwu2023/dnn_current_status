"""
Created on Wednesday April 27 2022
@author: qiangwu
"""
import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    """
        Generate n random numbers from Uniform(u1,u2)
    """
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_I(n, corr, Beta):
    """
        Generate data for Case 4
        Case 4 (Deep 2 Model): 
        g(X)={\sqrt(X1X2)/5+X3^2X4/4+\log(X4+1)/3+\exp(X5)/2}^2/5-C
        generate Case 1 interval-censored data
        Lambda_0(t)= \sqrt(t)/5
        Lambda(t) = Lambda_0(t)*exp(Z*beta+g(X))
        S(t) = \exp[-Lambda_0(t)*exp(Z*beta+g(X))]
        F(t) = 1-S(t)
        Lambda(t) = -log(S(t))/exp(Z*beta+g(X))
    """
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    X = ndm.multivariate_normal(mean, cov, n)
    # Constrain X to [0,2]
    X = np.clip(X, 0, 2)
    # ====The intercept term is approximately equal to 0.45 (by running choose_intercept.py)=====
    g_X = (np.sqrt(X[:,0]*X[:,1])/5 + X[:,2]**2*X[:,3]/4 + np.log(X[:,3]+1)/3 + np.exp(X[:,4])/2)**2/5 - 0.45
    Y = ndm.rand(n)
    T = (-5 * np.log(Y) * np.exp(-Z * Beta - g_X)) ** 2
    U = uniform_data(n, 0, 10)
    De = (T <= U)
    return {
        'Z': np.array(Z, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'De': np.array(De, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32')
    }

# Dat = generate_case_I(2000, 0.5, 1)
# g = Dat['g_X']
# np.max(g), np.min(g), np.mean(g), np.mean(Dat['De'])
