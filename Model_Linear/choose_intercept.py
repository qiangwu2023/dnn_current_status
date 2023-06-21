# =======Calculate the intercept item C=======
import numpy as np
import numpy.random as ndm
corr = 0.5
mean = np.zeros(5)
cov = np.identity(5) * (1-corr) + np.ones((5, 5)) * corr
X = ndm.multivariate_normal(mean, cov, 10000)
X = np.clip(X, 0, 2)
g_X = X[:,0]/2 + X[:,1]/3 + X[:,2]/4 + X[:,3]/5 + X[:,4]/6
C = np.mean(g_X)
C