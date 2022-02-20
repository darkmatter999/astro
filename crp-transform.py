#This function transforms (adds, subtracts) one relative orientation of a CRP representation into another
#Instead of using the usual DCM multiplication method, with CRPs we can transform using a more elegant direct way

import numpy as np
from mrp_conversion import convert_mrp

#First, we make implicit use of the fact that if we have, for instance, q [B/N], q [N/B] is simply the
#negative of it. In other words, we can simply implicitly negate the result in that case.

#Addition (given q [F/B] and q [B/N], find q [F/N])
#In other words, what is the relative attitude of spacecraft 1 to the inertial frame, if we have the relative
#attitude of spacecraft 1 to spacecraft 2 and the relative attitude of spacecraft 2 to the inertial frame?

#The inputs in both functions are two np arrays of 3x1 shape (two CRP sets)

def crpadd(q_fb, q_bn):
    q_fn = ((q_bn+q_fb)-np.cross(q_bn, q_fb))/(1-(np.dot(q_bn, q_fb)))
    return q_fn

#still doesn't work
def mrpadd(sigma_rb, sigma_bn):
    sigma_rn = ((1-np.square(np.linalg.norm(sigma_bn))*sigma_rb)-
    (1-np.square(np.linalg.norm(sigma_rb))*sigma_bn)-
    np.cross(2*sigma_rb, sigma_bn)) / ((1+(np.square(np.linalg.norm(sigma_bn))*np.square(np.linalg.norm(sigma_rb))))-np.dot(2*sigma_bn, sigma_rb))
    return convert_mrp(sigma_rn), sigma_rn

#Subtraction (given q [F/N] and q [B/N], find [F/B], in other words, the relative orientation of spacecraft 1
#to spacecraft 2)

def crpsub(q_fn, q_bn):
    q_fb = ((q_fn-q_bn)+np.cross(q_fn, q_bn))/(1+(np.dot(q_fn, q_bn)))
    return q_fb

#print (-crpsub(np.array([0.1, 0.2, 0.3]), np.array([-0.3, 0.3, 0.1])))

print (mrpadd(np.array([-0.1, 0.3, 0.1]), np.array([0.1, 0.2, 0.3])))