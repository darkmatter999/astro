import numpy as np 

#Static Attitude Determination through the Triad method
# 
#Problem statement:
#On a spacecraft, we measure the spacecraft's frame relative to the sun and also relative to
#the magnetic field (via sensors). Also, we possess knowledge of the inertial frame relative
#to sun and magnetic field. We know that one of the two measurements (usually the sun measurement)
#is more exact than the other.
#Given above information, we want to determine/estimate the spacecraft's attitude, i.e. we want
#to construct an adequate attitude representation. This could be either a DCM, or other attitude
#representation formats, such as quaternions, PRV, CRP or MRP.
#
#Using the triad method, we
#1. Form two sets of three Triad vectors: The Body Frame Triad Vectors and the Inertial Frame Triad Vectors.
#   The Triad frame must be orthogonal to Body Frame and Inertial Frame.
#2. Having these (B and N) Triad vectors, we construct two DCMs, BT and NT (i.e. Spacecraft
#   relative to the new Triad frame, inertial frame relative to Triad frame)
#3. Finally, we can construct BN (our attitude estimation) by composing BT and NT to form BN, by 
#   computing BT*NT.T

#Input parameters are the body-sun and body-magnetic field sensor measurements, and inertial-sun and inertial-m.f.
def triad_attitude_estimate(b_sun, b_mf, n_sun, n_mf):
    #define the more exact of the two body frame measurements as bt_1
    bt_1 = b_sun
    #compute bt_2 as the cross product of bt1 and the second (usually the magnetic field) measurement, over
    #the norm of that cross product
    bt_2 = (np.cross(bt_1, b_mf))/np.linalg.norm(np.cross(bt_1, b_mf))
    #bt_3 is simply the cross product of bt_1 and bt_2
    bt_3 = np.cross(bt_1, bt_2)

    #We form the inertial Triad vectors in a similar fashion
    nt_1 = n_sun
    nt_2 = (np.cross(nt_1, n_mf))/np.linalg.norm(np.cross(nt_1, n_mf))
    nt_3 = np.cross(nt_1, nt_2)

    #Constructing the DCMs BT and NT. bt_1, bt_2, ... are lined up in columns, hence the transpose
    BT = np.array([bt_1, bt_2, bt_3]).T
    NT = np.array([nt_1, nt_2, nt_3]).T

    #Compose BN by 'adding' BT and NT. Recall BN = BT*TN, i.e. BT*NT.T
    BN = np.matmul(BT, NT.T)

    return BN, np.matmul(BN,BN.T)


b_v1 = np.array([0.8273, 0.5541, -0.0920])
b_v2 = np.array([-0.8285, 0.5522, -0.0955])
n_v1 = np.array([-0.1517, -0.9669, 0.2050])
n_v2 = np.array([-0.8393, 0.4494, -0.3044])

print (triad_attitude_estimate(b_v1, b_v2, n_v1, n_v2))





