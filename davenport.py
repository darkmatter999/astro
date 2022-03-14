import numpy as np
from quat_to_dcm import quat_to_dcm

#One problem with the TRIAD method is that it takes into account only two measurements (two vectors to form the
#third one - the triad vector). These are usually the sun and magnetic field relative measurements.
#We could get a more exact attitude measurement if we had more observations, e.g. if we
#had multiple star trackers.
#Then we could solve Wahba's problem of minimizing a loss function J and so finding the most appropriate 
#attitude description. More specifically, instead of v1 and v2 for B and N, respectively, as for TRIAD, we now
#vk measurements for B and N, and need to integrate all these k observations to yield the optimal attitude 
#description. Then, if all measurements are perfect, J is 0, i.e. there is no error. Otherwise we keep on
#minimizing the loss (or, alternatively, maximizing the gain).
#Essentially, we 'control' the loss function with weights we choose based on how much we 'trust' a given
#observation, on how exact we think a given observation is. For instance, we could still weigh the sun
#measurement higher than a certain star measurement. If all observations are equally valid to us, we simply
#set all weights to 1.
#So we want to minimize:
#J[BN] = 1/2*(sum of observations)*wk(weights)*squared L2 Norm(B_vk - [BN]N_vk)

#Davenport's q method applied Wahba's problem to quaternions. Here we only need to deal with four parameters
#instead of 9 in the case of DCM. 
#Furthermore, by using the definitions of the norm, we can essentially rewrite above cost function as
#J = (sum of observations)wk*(1-B_vk.T*[BN]*N_vk)
#With the q method we maximize the gain instead of minimizing the loss, i.e. we simply take away the subtraction
#from 1 in above restated J function:
#G=(sum of observations)wk*(B_vk.T*[BN]*N_vk)
#We want to make G as big as possible, which we can do by finding the appropriate [BN] description.
#Using quaternions, we can then rewrite G as:
#G(beta) = beta.T*[K]*beta, where K is a composite 4x4 matrix, composed of manipulations around the the matrix
#B, which is the sum of the (weighted) outer products of b_vk and n_vk.
#(The individual 4 components of K will be introduced in below code)

#Now, what then is the appropriate quaternion attitude description? To find this, we need to take the
#Eigensystem (eigenvalues and eigenvectors) of the 4x4 matrix K. The correct description 
#is then the eigenvector associated with the **largest** eigenvalue of the 4 resulting eigenvalues.
#It turns out that the aforementioned cost function maximization G is simply that largest eigenvalue,
#hence our solution.
#It is however important to choose the 'right' quaternion set since eigenvectors are not unique. 
#Numpy (and MATLAB, Octave, for that matter) will give us one of two valid eigenvectors.
#We cannot influence which. We need to choose the positive eigenvector in order to choose the 'short rotation' 
#and thus flip signs if necessary.

#Davenport's q method is a bit out of fashion nowadays and superseded by faster algorithms since the 
#Eigensystem computation is rather computationally expensive, and so the q method doesn't scale well
#with higher-dimension systems.

def davenport_attitude(m_b, m_n, weights): 
    #m_b and m_n are lists of observations (as described, we do not necessarily only have 2 observations here!)
    #weights is a list of the corresponding weights
    B = np.zeros((3,3))
    for (i, j, k) in zip(m_b, m_n, weights):
        B = B + (k*np.outer(i/np.linalg.norm(i),j/np.linalg.norm(j)))
    #Compose K
    #First, sigma is the trace of B, a scalar
    sigma = np.trace(B)
    #To get Z, we subtract B's non-diagonal elements from each other
    #The result is a 3x1 vector
    Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])
    #Z_transpose is simply the transpose of Z
    Z_transpose = Z.T
    #Finally, S is yielded by adding B with B transposed
    S = B + B.T
    #Having these 4 components (sigma, Z, Z_transpose and S) we can compose K
    K = np.array([[sigma, Z_transpose[0], Z_transpose[1], Z_transpose[2]],
    [Z[0], S[0,0]-sigma, S[0,1], S[0,2]],
    [Z[1], S[1,0], S[1,1]-sigma, S[1,2]],
    [Z[2], S[2,0], S[2,1], S[2,2]-sigma]])
    #Compute the eigensystem of K
    eigensystem = np.linalg.eig(K)
    #Extract the index of the largest eigenvalue
    largest_eigenvalue_idx = np.ndarray.tolist(eigensystem[0]).index(max(eigensystem[0]))
    #Extract the resulting quaternion attitude representation which is the eigenvector associated with the
    #largest eigenvalue of K
    BN_quat = eigensystem[1][:,largest_eigenvalue_idx]
    #Convert the resulting quaternion set to DCM format
    BN = quat_to_dcm(BN_quat)

    return BN


b = [np.array([0.8273, 0.5541, -0.0920]), np.array([-0.8285, 0.5522, -0.0955])]
n = [np.array([-0.1517, -0.9669, 0.2050]), np.array([-0.8393, 0.4494, -0.3044])]
w = [1,1]

print (davenport_attitude(b, n, w))








