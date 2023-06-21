import numpy as np
def B_spline_basis(i, p, u, nodevec):
    if p == 0:
        if (nodevec[i] <= u) and (u < nodevec[i + 1]):
            result = 1
        else:
            result = 0
    else:
        length1 = nodevec[i + p] - nodevec[i]
        length2 = nodevec[i + p + 1] - nodevec[i + 1]
        if length1 == 0:
            alpha = 0
        else:
            alpha = (u - nodevec[i]) / length1
        if length2 == 0:
            beta = 0
        else:
            beta = (nodevec[i + p + 1] - u) / length2
        # Recursion
        result = alpha * B_spline_basis(i, p - 1, u, nodevec) + beta * B_spline_basis(i + 1, p - 1, u, nodevec)
    return result

# This function returns the value of the jth cubic spline basis function at u on the incomplete interval (j has 6 choices)
def B_other_basis(j, m, u, nodevec):
    if (j==0):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = (nodevec[1]-u)**3/(nodevec[1]-nodevec[0])**3
        else:
            result = 0
        return result
    elif (j==1):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = (nodevec[2]-u)**2*(u-nodevec[0])/((nodevec[2]-nodevec[0])**2*(nodevec[1]-nodevec[0]))
        elif (nodevec[1]<=u) and (u<nodevec[2]):
            result = (nodevec[2]-u)**3/((nodevec[2]-nodevec[0])**2*(nodevec[2]-nodevec[1]))
        else:
            result = 0
        return result
    elif (j==2):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = ((nodevec[3]-u)*(u-nodevec[0])**2)/((nodevec[3]-nodevec[0])*(nodevec[2]-nodevec[0])*(nodevec[1]-nodevec[0]))
        elif (nodevec[1]<=u) and (u<nodevec[2]):
            result = ((nodevec[3]-u)*(u-nodevec[0])*(nodevec[2]-u))/((nodevec[3]-nodevec[0])*(nodevec[2]-nodevec[0])*(nodevec[2]-nodevec[1])) + (nodevec[3]-u)**2*(u-nodevec[1])/((nodevec[3]-nodevec[0])*(nodevec[3]-nodevec[1])*(nodevec[2]-nodevec[1]))
        elif (nodevec[2]<=u) and (u<nodevec[3]):
            result = (nodevec[3]-u)**3/((nodevec[3]-nodevec[0])*(nodevec[3]-nodevec[1])*(nodevec[3]-nodevec[2]))
        else:
            result = 0
        return result
    elif (j==3):
        if (nodevec[m-2]<=u) and (u<nodevec[m-1]):
            result = (u-nodevec[m-2])**3/((nodevec[m+1]-nodevec[m-2])*(nodevec[m]-nodevec[m-2])*(nodevec[m-1]-nodevec[m-2]))
        elif (nodevec[m-1]<=u) and (u<=nodevec[m]):
            result = (u-nodevec[m-2])**2*(nodevec[m]-u)/((nodevec[m+1]-nodevec[m-2])*(nodevec[m]-nodevec[m-2])*(nodevec[m]-nodevec[m-1])) + (u-nodevec[m-2])*(nodevec[m+1]-u)*(u-nodevec[m-1])/((nodevec[m+1]-nodevec[m-2])*(nodevec[m+1]-nodevec[m-1])*(nodevec[m]-nodevec[m-1]))
        elif (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = (u-nodevec[m-2])*(nodevec[m+1]-u)**2/((nodevec[m+1]-nodevec[m-2])*(nodevec[m+1]-nodevec[m-1])*(nodevec[m+1]-nodevec[m]))
        else:
            result = 0
        return result
    elif (j==4):
        if (nodevec[m-1]<=u) and (u<nodevec[m]):
            result = (u-nodevec[m-1])**3/((nodevec[m+1]-nodevec[m-1])**2*(nodevec[m]-nodevec[m-1]))
        elif (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = (u-nodevec[m-1])**2*(nodevec[m+1]-u)/((nodevec[m+1]-nodevec[m-1])**2*(nodevec[m+1]-nodevec[m]))
        else:
            result = 0
        return result
    elif (j==5):
        if (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = (u-nodevec[m])**3/(nodevec[m+1]-nodevec[m])**3
        else:
            result = 0
        return result
# Estimate the value of all spline basis functions at the point u, and get a vector with the same dimension (m+4) as the node parameter
def B_spline(m, u, nodevec):
    B_p = [] # dimension (m+4)
    # There are (m-2) cubic splines in the middle (the interval is complete)
    for i in range(m-2):
        B_p.append(B_spline_basis(i, 3, u, nodevec))
    # Three integral splines on each side (a total of 6 interval incomplete splines)
    for j in range(6):
        B_p.append(B_other_basis(j, m, u, nodevec))
    B_p = np.array(B_p, dtype='float32')
    return B_p


# Let u traverse U, n*(m+4) matrix
def B_S(m, U, nodevec):
    B_u = np.zeros(shape=(len(U),m+4))
    for b in range(len(U)):
        u = U[b]
        B_u[b] = B_spline(m, u, nodevec)
    B_u = np.array(B_u, dtype='float32')
    return B_u
