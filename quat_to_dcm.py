import numpy as np

#Converts a quaternion set to the DCM format

def quat_to_dcm(beta):
    C = np.array([[np.square(beta[0])+np.square(beta[1])-np.square(beta[2])-np.square(beta[3]), 
    2*((beta[1]*beta[2])+(beta[0]*beta[3])), 2*((beta[1]*beta[3])-(beta[0]*beta[2]))],
    [2*((beta[1]*beta[2])-(beta[0]*beta[3])), 
    np.square(beta[0])-np.square(beta[1])+np.square(beta[2])-np.square(beta[3]),
    2*((beta[2]*beta[3])+(beta[0]*beta[1]))], [2*((beta[1]*beta[3])+(beta[0]*beta[2])), 
    2*((beta[2]*beta[3])-(beta[0]*beta[1])),
    np.square(beta[0])-np.square(beta[1])-np.square(beta[2])+np.square(beta[3])]])

    return C

#print (quat_to_dcm(np.array([ 0.11335196, -0.83314496,  0.50296458, -0.20011858])))