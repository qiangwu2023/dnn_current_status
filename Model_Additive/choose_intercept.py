# =======Calculate the intercept item C=======
import numpy as np
import numpy.random as ndm
corr = 0.5
mean = np.zeros(5)
cov = np.identity(5) * (1-corr) + np.ones((5, 5)) * corr
X = ndm.multivariate_normal(mean, cov, 10000)
X = np.clip(X, 0, 2)
g_X = X[:,0]**2/2 + 2*np.log(X[:,1]+1)/5 + 3*np.sqrt(X[:,2])/10 + np.exp(X[:,3])/5 + X[:,4]**3/10
C = np.mean(g_X)
C