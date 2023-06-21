# =======Calculate the intercept item C=======
import numpy as np
import numpy.random as ndm
corr = 0.5
mean = np.zeros(5)
cov = np.identity(5) * (1-corr) + np.ones((5, 5)) * corr
X = ndm.multivariate_normal(mean, cov, 10000)
X = np.clip(X, 0, 2)
g_X = np.sqrt(X[:,0]*X[:,1])/5 + X[:,2]**2*X[:,3]/4 + np.log(X[:,3]+1)/3 + np.exp(X[:,4])/2
C = np.mean(g_X)
C