import numpy as np
# Calculate the integral spline, i is the control vertex number, p is the degree, u is the substituted value, and NodeVector is the node vector
# How is the node vector NodeVector determined?
# There are m+p p-time integral spline basis functions for m inner nodes (monotonically increasing)
# Consider here -------p=3----------
# Then write a cubic integral spline (integrate on the basis of quadratic basis spline)
# This function returns the value of the i-th cubic integral spline basis function at u on the complete interval (i has m-1 choices)
def inte_basis(i, u, nodevec):
    if (nodevec[i] <= u) and (u < nodevec[i+1]):
        result = (u-nodevec[i])**3/(3*(nodevec[i+2]-nodevec[i])*(nodevec[i+1]-nodevec[i]))
    elif (nodevec[i+1] <= u) and (u < nodevec[i+2]):
        result = (nodevec[i+1]-nodevec[i])**2/(3*(nodevec[i+2]-nodevec[i])) + (-u**3/3+u**2*(nodevec[i+2]+nodevec[i])/2-nodevec[i+2]*nodevec[i]*u+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+2]+nodevec[i])/2+nodevec[i+2]*nodevec[i]*nodevec[i+1])/((nodevec[i+2]-nodevec[i])*(nodevec[i+2]-nodevec[i+1])) + (-u**3/3+u**2*(nodevec[i+3]+nodevec[i+1])/2-nodevec[i+3]*nodevec[i+1]*u+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+3]+nodevec[i+1])/2+nodevec[i+3]*nodevec[i+1]*nodevec[i+1])/((nodevec[i+3]-nodevec[i+1])*(nodevec[i+2]-nodevec[i+1]))
    elif (nodevec[i+2] <= u) and (u < nodevec[i+3]):
        result = (nodevec[i+1]-nodevec[i])**2/(3*(nodevec[i+2]-nodevec[i])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+2]+nodevec[i])/2-nodevec[i+2]*nodevec[i]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+2]+nodevec[i])/2+nodevec[i+2]*nodevec[i]*nodevec[i+1])/((nodevec[i+2]-nodevec[i])*(nodevec[i+2]-nodevec[i+1])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+3]+nodevec[i+1])/2-nodevec[i+3]*nodevec[i+1]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+3]+nodevec[i+1])/2+nodevec[i+3]*nodevec[i+1]*nodevec[i+1])/((nodevec[i+3]-nodevec[i+1])*(nodevec[i+2]-nodevec[i+1])) + ((u-nodevec[i+3])**3+(nodevec[i+3]-nodevec[i+2])**3)/(3*(nodevec[i+3]-nodevec[i+1])*(nodevec[i+3]-nodevec[i+2]))
    elif (nodevec[i+3] <= u):
        result = (nodevec[i+1]-nodevec[i])**2/(3*(nodevec[i+2]-nodevec[i])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+2]+nodevec[i])/2-nodevec[i+2]*nodevec[i]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+2]+nodevec[i])/2+nodevec[i+2]*nodevec[i]*nodevec[i+1])/((nodevec[i+2]-nodevec[i])*(nodevec[i+2]-nodevec[i+1])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+3]+nodevec[i+1])/2-nodevec[i+3]*nodevec[i+1]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+3]+nodevec[i+1])/2+nodevec[i+3]*nodevec[i+1]*nodevec[i+1])/((nodevec[i+3]-nodevec[i+1])*(nodevec[i+2]-nodevec[i+1])) + (nodevec[i+3]-nodevec[i+2])**3/(3*(nodevec[i+3]-nodevec[i+1])*(nodevec[i+3]-nodevec[i+2]))
    else:
        result = 0
    return result

# This function returns the value of the i-th cubic integral spline basis function at u on an incomplete interval (i has 4 choices)
def Ic_inte_basis(j, m, u, nodevec):
    if (j==0):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = 3*((nodevec[1]-nodevec[0])**3-(nodevec[1]-u)**3)/(3*(nodevec[1]-nodevec[0])**2)/(m+1)
        else:
            result = 3*(nodevec[1]-nodevec[0])/3/(m+1)
        return result
    elif (j==1):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = 3/3*(((-u**3/3+(u**2*(nodevec[0]+nodevec[2])/2-nodevec[0]*nodevec[2]*u))-(-nodevec[0]**3/3+(nodevec[0]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]**2*nodevec[2])))/((nodevec[1]-nodevec[0])*(nodevec[2]-nodevec[1])))/(m+1)
        elif (nodevec[1]<=u) and (u<nodevec[2]):
            result = 3/3*(((-nodevec[1]**3/3+(nodevec[1]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]*nodevec[2]*nodevec[1]))-(-nodevec[0]**3/3+(nodevec[0]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]**2*nodevec[2])))/((nodevec[1]-nodevec[0])*(nodevec[2]-nodevec[1]))+((nodevec[2]-nodevec[1])**3-(nodevec[2]-u)**3)/(3*(nodevec[2]-nodevec[1])**2))/(m+1)
        else:
            result = 3/3*(((-nodevec[1]**3/3+(nodevec[1]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]*nodevec[2]*nodevec[1]))-(-nodevec[0]**3/3+(nodevec[0]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]**2*nodevec[2])))/((nodevec[1]-nodevec[0])*(nodevec[2]-nodevec[1]))+(nodevec[2]-nodevec[1])/3)/(m+1)
        return result
    elif (j==2):
        if (nodevec[m-1]<=u) and (u<nodevec[m]):
            result = 3/3*((u-nodevec[m-1])**3/(3*(nodevec[m]-nodevec[m-1])**2))/2
        elif (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = 3/3*((nodevec[m]-nodevec[m-1])/3+((-u**3/3+(u**2*(nodevec[m-1]+nodevec[m+1])/2-nodevec[m-1]*nodevec[m+1]*u))-(-nodevec[m]**3/3+(nodevec[m]**2*(nodevec[m-1]+nodevec[m+1])/2-nodevec[m-1]*nodevec[m+1]*nodevec[m])))/((nodevec[m]-nodevec[m-1])*(nodevec[m+1]-nodevec[m])))/2
        else:
            result = 0
        return result
    elif (j==3):
        if (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = 3*(u-nodevec[m])**3/(3*(nodevec[m+1]-nodevec[m])**2)
        else:
            result = 0
        return result

# Estimate the value of all spline basis functions at the point u, and get a vector with the same dimension (m+3) as the node parameters
def I_spline(m, u, nodevec):
    B_p = [] # m+3
    # There are m-1 cubic integral splines in the middle (the interval is complete)
    for i in range(m-1):
        B_p.append(inte_basis(i, u, nodevec))
    # Two integral splines on each side (a total of 4 integral splines with incomplete intervals)
    for j in range(4):
        B_p.append(Ic_inte_basis(j, m, u, nodevec))
    B_p = np.array(B_p, dtype='float32')
    return B_p


def I_U(m, U, nodevec):
    I_u = np.zeros(shape=(len(U),m+3))
    for b in range(len(U)):
        u = U[b]
        I_u[b] = I_spline(m, u, nodevec)
    I_u = np.array(I_u, dtype='float32')
    # # Let u traverse U, n*(m+3) matrix
    return I_u


# Define the vector value of the p-degree integral spline of a function at a vector U
def I_S(m, c0, U, nodevec):
    B_value = []
    for b in range(len(U)):
        u = U[b]
        B_value.append(np.sum(I_spline(m, u, nodevec)*c0))
    B_value = np.array(B_value, dtype='float32')
    return B_value
