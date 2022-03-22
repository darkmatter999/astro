import numpy as np
from mrp_cayley import skew_transform
from crpdcm import crptodcm

#OLAE, the Optimal Linear Attitude Estimator, other than Davenport's q-method and QUEST, doesn't solve Wahba's
#problem, but solves the attitude estimation problem in a rigorously linear way by making use of the features
#of the Cayley transform.

#It is a rather new technique, developed around 10 years ago.

#As with Davenport and QUEST, we have our body and inertial measurements, and the related weights, as input
#parameters again.

def olae(m_n, m_b, weights):
    S = np.zeros([len(m_n), 3*3]) #initialize S
    D = np.zeros([len(m_n), 3]) #initialize d
    #S is the skew-symmetric form of the iterated sum of the body and inertial measurements of each 
    #individual (i-th) measurement. That means each 3x1 measurement is first being converted into
    #a 3x3 skew symmetric matrix. With two measurements, we'd have two 3x3 matrices. The final shape
    #of S is then converted into (n*3, 3).
    for (a, b, i) in zip(m_b, m_n, range(len(S))):
        S[i] = np.reshape(skew_transform(a + b), ((1,9)))
    S = np.reshape(S, ((len(m_n)*3), 3))

    #D subtracts the inertial from the body measurement values
    #instead of adding them. It is in nx1 vector form.
    #Important: The matrices and vectors in OLAE do not contain sensible geometric insights. They are
    #essentially just used for doing the matrix math necessary for the Cayley Transform - based calculations.
    for (c, d, j) in zip(m_b, m_n, range(len(D))):
        D[j] = c - d
    D = np.reshape(D, ((len(m_n)*3), 1))

    #'weightless' - We could 'hardcode' 6 weights here and compose an 'identity-like' matrix with the
    #weights multiplied with each '1' entry, or use a loop to compose such a matrix with random weights.
    #For now, let's keep it weightless, i.e. all weights being exactly 1.
    W = np.eye(len(m_n)*3) 

    #This is the classical Cayley-Transform equation (remember: works both ways for CRP!)
    c1 = np.linalg.inv(np.dot(np.dot(S.T, W), S))
    c2 = np.dot(np.dot(S.T, W), D)
    q = np.reshape(np.dot(c1, c2), ((3,))) #find the optimal CRP

    BN = crptodcm(q).T #find (convert to) optimal DCM

    return BN


b = [np.array([0.8273, 0.5541, -0.0920]), np.array([-0.8285, 0.5522, -0.0955])]
n = [np.array([-0.1517, -0.9669, 0.2050]), np.array([-0.8393, 0.4494, -0.3044])]
w = [1,1] #The weight input parameter, as explained above, is not used here. TBD: Implement a 'weighted version'

print (olae(b, n, w))


