import numpy as np
import math

#Given a 'true' BN DCM, we want to check how good our attitude estimation is, i.e. what the error between
#the true values and the estimated values is.
#To do this, we take the dot product of our BN estimation and the transposed true BN, which should - ideally,
#in the absence of any error - just give us back the identity matrix (recall [C][C].T = I).
#To find a convenient error measure, we convert this 'near-identity' matrix to a Principal Rotation Vector, and
#take the norm of that resulting PRV. Finally, convert the norm to degrees to establish how far we're 'off'
#in terms of degrees.
#In real-life application, this can be an iterative process of establishing convergence.

def triad_accuracy(BN_estimated, BN_true):
    accuracy_matrix = np.dot(BN_estimated, BN_true.T) #This is the 'near-identity' matrix
    #First, find Principal Rotation Angle
    phi = np.arccos(0.5*(np.trace(accuracy_matrix)-1))
    #Extract the Principal Rotation Vector e_hat
    e_hat = (phi/2/np.sin(phi))*np.array([accuracy_matrix[1,2]-accuracy_matrix[2,1], accuracy_matrix[2,0]-accuracy_matrix[0,2], accuracy_matrix[0,1]-accuracy_matrix[1,0]])
    #Extract the norm of e_hat
    norm_e_hat = np.linalg.norm(e_hat)
    #Convert result to degrees
    error = np.rad2deg(norm_e_hat)

    return error

BN_e = np.array([[0.969846, 0.17101, 0.173648], [-0.200706, 0.96461, 0.17101], [-0.138258, -0.200706, 0.969846]])
BN_t = np.array([[0.963592, 0.187303, 0.190809], [-0.223042, 0.956645, 0.187303], [-0.147454, -0.223042, 0.963592]])

print (triad_accuracy(BN_e, BN_t))

C = np.array([[ 0.99970013, -0.02019574, -0.01382426],
       [ 0.0199059 ,  0.99958589, -0.02076977],
       [ 0.01423821,  0.02048801,  0.99968841]])
def C2PRV(C):
    """
    C2PRV

    	Q = C2PRV(C) translates the 3x3 direction cosine matrix
    	C into the corresponding 3x1 principal rotation vector Q,
    	where the first component of Q is the principal rotation angle
    	phi (0<= phi <= Pi)
    """

    cp = (np.trace(C)-1)/2
    p = math.acos(cp)
    sp = p/2/math.sin(p)
    q = np.matrix('0.;0.;0.')
    q[0,0] = (C[1,2]-C[2,1])*sp
    q[1,0] = (C[2,0]-C[0,2])*sp
    q[2,0] = (C[0,1]-C[1,0])*sp

    return q
    

print (C2PRV(C))





