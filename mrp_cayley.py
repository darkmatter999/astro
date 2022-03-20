#This set of functions converts a set of MRP to DCM format and vice versa

import numpy as np
from scipy.linalg import sqrtm

#First, we can transform any 3x1 vector to its skew-symmetric matrix form by
def skew_transform(v):
    M = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return M

#Now, given a set of MRP, use the Cayley Transform to get the corresponding DCM representation
def cayley_mrp_to_dcm(sigma):
    S = skew_transform(sigma)
    C = np.matmul(np.linalg.matrix_power(np.eye(3)-S, 2), np.linalg.matrix_power(np.eye(3)+S, -2))
    return C

#For the inverse transformation (DCM --> MRP) - other than with CRP - we need the square root of C to
#do the Cayley Transform

def cayley_dcm_to_mrp(C):
    W = sqrtm(C) #W actually is a 'half rotation', i.e. W multiplied with itself gives us C, hence the full rotation
    S = np.matmul(np.eye(3)-W, np.linalg.inv(np.eye(3)+W)) 
    #extract the sigma (MRP) vector from the skew-symmetric matrix S
    S_flattened = np.ndarray.flatten(S) #prior to extraction, for easier indexing, first flatten S
    #get rid of the imaginary part of the sqrtm operation by converting to float
    sigma = np.array([float(S_flattened[7]), float(S_flattened[2]), float(S_flattened[3])])
    return sigma

#regarding CC19
#res = np.matmul(cayley_mrp_to_dcm(np.array([-0.1, 0.3, 0.1])), cayley_mrp_to_dcm(np.array([0.1, 0.2, 0.3])))
#print (cayley_dcm_to_mrp(res))
