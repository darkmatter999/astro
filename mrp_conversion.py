import numpy as np

#This function converts a given MRP sigma vector to its counterpart. The first possible conversion is from
#a 'short' rotation to a 'long' one (to the 'shadow' set), the second is vice versa.
#We can identify a short rotation if the norm of sigma (the MRP vector) <= 1, and a long rotation
#if the norm >= 1.

#For higher-dimensional MRP sets a for-loop could be implemented
def convert_mrp(sigma):
    sigma_1 = sigma[0]
    sigma_2 = sigma[1]
    sigma_3 = sigma[2]
    sigma_squared = np.dot(sigma, sigma)

    sigma_shadow_1 = -sigma_1/sigma_squared
    sigma_shadow_2 = -sigma_2/sigma_squared
    sigma_shadow_3 = -sigma_3/sigma_squared

    #get the 'shadow' set of MRP, describing the counter-rotation
    sigma_shadow = np.array([sigma_shadow_1, sigma_shadow_2, sigma_shadow_3]) 

    #Return the converted set and check what rotation (short or long) its norm conforms to
    return sigma_shadow, 'short rotation' if np.linalg.norm(sigma_shadow) < 1 else 'long rotation', np.linalg.norm(sigma_shadow)

print (convert_mrp(np.array([2.17792845, 2.16974515, 2.40004676])))





