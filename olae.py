import numpy as np
from mrp_cayley import skew_transform

#OLAE, the Optimal Linear Attitude Estimator, other than Davenport's q-method and QUEST, doesn't solve Wahba's
#problem, but solves the attitude estimation problem in a rigorously linear way by making use of the features
#of the Cayley transform.

#It is a rather new technique, developed around 10 years ago.

#As with Davenport and QUEST, we have our body and inertial measurements, and the related weights, as input
#parameters again.

def olae(m_n, m_b, weights):
    S = np.zeros([len(m_n), 3]) #initialize s
    D = np.zeros([len(m_n), 3]) #initialize d
    #S is the iterated sum of the body and inertial measurements of each individual (i-th) measurement
    for (a, b, i) in zip(m_b, m_n, range(len(S))):
        S[i] = a + b
    S_tilde = np.zeros([len(m_n)*3, 3]) #implement skew-symmetric (6x3) matrix form

    #Do the same for D, the only difference is that D subtracts the inertial from the body measurement values
    #instead of adding them
    for (c, d, j) in zip(m_b, m_n, range(len(D))):
        D[j] = c - d
    D = np.reshape(D, ((len(m_n)*3), 1))
    
    return D


b = [np.array([0.8273, 0.5541, -0.0920]), np.array([-0.8285, 0.5522, -0.0955])]
n = [np.array([-0.1517, -0.9669, 0.2050]), np.array([-0.8393, 0.4494, -0.3044])]
w = [1,1]

print (olae(b, n, w))

