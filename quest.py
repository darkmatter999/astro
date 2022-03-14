import numpy as np
from scipy import optimize
from crpdcm import crptodcm

#The QUEST method, developed by Malcolm Shuster, yields the exact same results as Davenport's q-Method, but is 
#computationally more efficient, since it is not necessary to solve an eigenvalue/eigenvector system.
#Instead of this, we need to do iterative (Newton-Raphson) root-solving, which is, however, quicker and thus
#'cheaper'.
#The main intuition behind QUEST is that the largest eigenvalue we were required to find with Davenport can in
#fact be quite precisely estimated just by taking the sum of the weights (e.g. 1 + 1 = 2, if we have two weights
#and both are 1, then the largest eigenvalue should be reasonably close to 2). Just taking the sum of the
#weights, however, is not precise enough the more weights we deal with. Here, taking a first good guess (i.e.
#the mentioned weight sum) and then iterative Newton-Raphson solving for the precise largest eigenvalue is used.

#Specifically, in order to solve for the optimal attitude, we must iteratively solve
#f(s) = det([K] - s[I4x4]) = 0. (K is known from Davenport already)
#Interestingly, QUEST (other than Davenport) doesn't yield a quaternion set, but CRP. (In order to get the
#quaternions, we'd need to convert from CRP to quaternions, but of course we can also convert directly from
#CRP to DCM).
#So, once we've found the largest eigenvalue, we use the already known elements from Davenport to solve directly
#for the optimal attitude in CRP:
#q_dash = ((lambda_opt (the largest eigenvalue) + sigma)[I3x3] - [S])⁻¹ [Z]

#The first steps of QUEST are identical to Davenport
def quest_attitude(m_b, m_n, weights): 
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
    #Here's where the crucial difference between Davenport and QUEST comes in:
    #Equipped with K, we can iteratively solve for the largest eigenvalue (notably again, without solving
    #a complete Eigensystem, making the computation considerably cheaper.)
    #First, we define the function to solve iteratively with Newton-Raphson. We use the scipy.optimize framework
    #for this.
    def f(s):
        return (np.linalg.det((K) - s*np.eye(4)))
    lambda_opt = optimize.newton(f, sum(weights))
    #Having lambda_opt, we can solve for the optimal CRP.
    q_dash = np.dot(np.linalg.inv((lambda_opt + sigma)*np.eye(3)-S),Z)
    #Finally we solve for the optimal attitude DCM by converting the CRP accordingly.
    #(We might also seek the optimal quaternion set as well, but this is not necessary if our goal is
    #to yield a DCM as final answer)
    BN = crptodcm(q_dash)

    return BN

b = [np.array([0.8273, 0.5541, -0.0920]), np.array([-0.8285, 0.5522, -0.0955])]
n = [np.array([-0.1517, -0.9669, 0.2050]), np.array([-0.8393, 0.4494, -0.3044])]
w = [1,1]

print (quest_attitude(b, n, w))

