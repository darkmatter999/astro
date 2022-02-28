import numpy as np
from dcm_to_prv import dcm_to_prv

#Given a 'true' BN DCM, we want to check how good our attitude estimation is, i.e. what the error between
#the true values and the estimated values is.
#To do this, we take the dot product of our BN estimation and the transposed true BN, which should - ideally,
#in the absence of any error - just give us back the identity matrix (recall [C][C].T = I).
#To find a convenient error measure, we convert this 'near-identity' matrix to a Principal Rotation Vector, and
#take the norm of that resulting PRV. Finally, convert the norm to degrees to establish how far we're 'off'
#in terms of degrees.
#In real-life application, this can be an iterative process of establishing convergence (e.g. through loss functions)
#Here, we simply 'make up' (simulate) a true BN matrix.

def triad_accuracy(BN_estimated, BN_true):
    #This is the 'near-identity' matrix.
    #If BN_estimated and BN_true were exactly the same, we would simply get back the identity matrix
    accuracy_matrix = np.dot(BN_estimated, BN_true.T) 
    #First, find the Principal Rotation Angle and extract the Principal Rotation Vector gamma
    gamma = dcm_to_prv(accuracy_matrix)
    #Extract the norm of e_hat, which will simply give us back phi (see dcm_to_prv method)
    #The full gamma extraction step could be sidestepped here, with only 
    #phi (the principal rotation angle) being required
    norm_gamma = np.linalg.norm(gamma)
    #For more convenient handling, convert result to degrees
    error = np.rad2deg(norm_gamma)

    return error

BN_e = np.array([[0.969846, 0.17101, 0.173648], [-0.200706, 0.96461, 0.17101], [-0.138258, -0.200706, 0.969846]])
BN_t = np.array([[0.963592, 0.187303, 0.190809], [-0.223042, 0.956645, 0.187303], [-0.147454, -0.223042, 0.963592]])

print (triad_accuracy(BN_e, BN_t))







