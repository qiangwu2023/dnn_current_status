# =======计算截距项C(提前算好带进去)=======
import numpy as np
import numpy.random as ndm
corr = 0.5
mean = np.zeros(5)
cov = np.identity(5) * (1-corr) + np.ones((5, 5)) * corr
X = ndm.multivariate_normal(mean, cov, 10000)
# X = 4/(1+np.exp(-X))-2
X = np.clip(X, 0, 2)
# g_X0 = X[:, 0]**2 + 2*X[:, 1]**2 + X[:, 2]**3 + np.sqrt(X[:, 3]+1)+ np.log(X[:, 4]+1)
# 验算Zhong文章中的那个均值是否正确
g_X = (np.sqrt(X[:,0]*X[:,1])/5 + X[:,2]**2*X[:,3]/4 + np.log(X[:,3]+1)/3 + np.exp(X[:,4])/2)**2/5
np.mean(g_X)