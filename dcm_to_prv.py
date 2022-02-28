import numpy as np

#Converts a DCM to the principal rotation axis e_hat and the principal rotation vector gamma

def dcm_to_prv(C):
    phi = np.arccos(0.5*(np.trace(C)-1))
    e_hat = (1/(2*np.sin(phi)))*np.array([C[1,2]-C[2,1], C[2,0]-C[0,2], C[0,1]-C[1,0]])
    #gamma is the Principal Rotation Vector (axis*angle). Its norm (magnitude) is simply phi.
    gamma = e_hat*phi
    return gamma

def norm_gamma(C, gamma):
    #Show that the norm/magnitude of gamma is simply the principal rotation angle which was extracted as the
    #first step of the DCM-->PRV conversion process
    phi = np.arccos(0.5*(np.trace(C)-1))
    norm_gamma = np.linalg.norm(gamma)
    #We round phi and the calculated norm since the decimal point evaluation doesn't yield the exact same 
    #results after 5 or so decimal points
    return round(phi, 4) == round(norm_gamma, 4)

DCM = np.array([[0.925417, 0.336824, 0.173648], [0.0296956, -0.521281, 0.852869], [0.377786, -0.784102, -0.492404]])

print (norm_gamma(DCM, dcm_to_prv(DCM)))



