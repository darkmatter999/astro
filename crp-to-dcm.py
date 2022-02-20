#This function converts a given set of Classical Rodrigues Parameters to a DCM

import numpy as np

def crptodcm(q):
    return np.dot(1/(1+(np.dot(q.T,q))),
    np.array([[1+np.square(q[0])-np.square(q[1])-np.square(q[2]),
    2*((q[0]*q[1])+q[2]), 2*((q[0]*q[2])-q[1])],
    [2*((q[1]*q[0])-q[2]), 1-np.square(q[0])+np.square(q[1])-np.square(q[2]), 2*((q[1]*q[2])+q[0])],
    [2*((q[2]*q[0])+q[1]), 2*((q[2]*q[1])-q[0]), 1-np.square(q[0])-np.square(q[1])+np.square(q[2])]]))

print (crptodcm(np.array([0.1, 0.2, 0.3])))

#CRP-->DCM conversion using the Cayley Transform Method --> returns the same DCM transformation as above
#classical method
#Here, we follow the Cayley Transform equation 
#C = (I - Q) (I + Q)⁻¹
#where I is the identity matrix and Q the skew-symmetric matrix representation of the three-parameter CRP set

#Thus, while both methods can be used, it is remarkable how elegant and 'simple' the Cayley Method is compared
#to the above sophisticated transformation method.

def cayley_crptodcm(q):
    Q = np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]) #skew-symmetric representation of q
    #the Cayley transform works with higher (N-) dimensions as well, so one could also parametrize below 'np.eye'
    #and above skew-symmetry conversion
    C = np.matmul(np.eye(3)-Q, np.linalg.inv(np.eye(3)+Q)) 
    return C

print (cayley_crptodcm(np.array([0.1, 0.2, 0.3])))



