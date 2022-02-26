'''
Given an initial attitude parametrization provided by an Euler angle set of three successive (yaw-pitch-roll, i.e.
3-2-1) rotations, and a provided instantaneous body angular velocity vector (omega), find the attitude of the 
spacecraft after a given time, through integrating the kinematic differential equation, which is composed of the
given Euler angles and the given omega vector.

Omega, usually measured by the rate gyro of the spacecraft, is the instantaneous angular velocity vector of 
frame B (body frame) relative to N (inertial frame). To solve the kinematic differential equation, we have to relate
the Euler angles (which do NOT represent a vector in the mathematical sense) to the omega vector (which IS a real
vector, comprised of magnitude x direction). We do this by multiplying the (inverse of the matrix representation
of the) mappings of the Euler successive rotations to the B frame with the respective omega components
(omega 1, 2, 3). Thus we find the instantaneous rates of change of the Euler angle set components.

We compose the 3x3 matrix in the function f by setting the three successive body rotations relative to the
B frame (translate into B frame components). To do this, we express the b unit vector components which are
modified through the successive rotations to the (b1, b2, b3) vectors.
For example, after the pitch rotation about the second angle, we get a new b2 component, let's call it b2'. We
need to express this new b2' in terms of B (b1,b2,b3) (map it to B). In other words, after 
the pitch (2 axis) rotation, where is b2' relative to the B frame components.
This way, we can use the canonical formula for finding the omega vector:
omega = omega1*b1+omega2*b2+omega3*b3, just by setting the derivatives of our three Euler angles as omega1, 2 and 3,
respectively, and the aforementioned transformed b coordinates as b1, 2, and 3, respectively.
We 'linearize' these three coordinate transformations by constructing a 3x3 matrix (NOT a DCM, just similar) 
made up of these three respective components.
For a 3-2-1 Euler angle set this results in a matrix as depicted in function f:

[[-np.sin(y[1]), 0, 1], [np.sin(y[2])*np.cos(y[1]), np.cos(y[2]), 0], [np.cos(y[2])*np.cos(y[1]), -np.sin(y[2]), 0]]

where y[1] and y[2] represent the pitch and roll components of the resulting 'y' Euler angle set. We don't need to
manipulate the yaw component since it is the first rotation and therefore we can use the (1 0 0) identity.

Finding the instantaneous angular velocity vector (which is a proper vector, direction x magnitude) now requires
multiplying the rates of change of the Euler angles with the body transformations (which, as stated above, are
represented in the compact 3x3 matrix format). 

Here, we already have omega, which then enables us to extract the instantaneous rates of change of the Euler
angles, hence of the spacecraft attitude. This is done by the Euler angles kinematic differential equation which
is the inverse of the kinematic equation for finding omega (essentially omega1*b1+omega2*b2+omega3*b3). 
Therefore we take the dot product of the inverted matrix with the coordinate transformation and the (provided) 
omega vector.

Then, given the initial set of Euler angles, we can integrate the resulting 'yawdot', 'pitchdot' and 'rolldot' to
find the new attitude after a certain time (42 seconds in below case).

The known singularity problem manifests itself in the inversion of the 3x3 mapping matri, since here we need to 
multiply the matrix and the omega vector by 1/cos(pitch component, i.e. 2nd angle). Setting the pitch angle to 90° 
results in a division by zero.
'''

#***********************************************************************************************************
#                 IMPORTANT: THIS CODE REFERS ONLY TO 3-2-1 EULER ANGLE REPRESENTATIONS
#***********************************************************************************************************

import numpy as np
from scipy.integrate import odeint
from mrp_conversion import convert_mrp

#***********************************************************
#Euler angle integration using the built-in scipy integrator
#***********************************************************

#define the function (differential kinematic equation) to be integrated
def f(y, t):
    #details of below function composition are described above
    B_map = np.array([[-np.sin(y[1]), 0, 1], [np.sin(y[2])*np.cos(y[1]), np.cos(y[2]), 0], [np.cos(y[2])*np.cos(y[1]), -np.sin(y[2]), 0]])
    B_inv = np.linalg.inv(B_map)
    b_omega = np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])*(20*np.pi/180)
    #Here, we have omega already given.
    #we need to use the inverted B_map (i.e. B_inv) in order to get the kinematic differential equation, giving
    #us the rates of change of the Euler rotations.
    return np.dot(B_inv, b_omega)

#define the initial attitude
y0 = np.array([40*np.pi/180, 30*np.pi/180, 80*np.pi/180])
#define the time steps to integrate - we integrate for 60 timesteps (60 seconds)
t = np.arange(0,60) #this is a vector containing the integers from 0 down to 59
#do the actual integration using the built-in scipy ODE integrator
y = odeint(f, y0, t)

#print (y) #complete matrix of the vectors for each timestep
#We want the attitude after 42 seconds
#print (np.sqrt(y[42][0]**2+y[42][1]**2+y[42][2]**2)) #elemental Euler angle normalization
#print (np.linalg.norm(y[42])) #Euler angle normalization using np.linalg.norm

#*********************************
#self-built Euler angle integrator
#*********************************

def euler_integrate(y0, t_end):
    y = y0 #define initial Euler attitude parametrization
    t = 0 #initial time is 0
    #make the time step as small as possible to get a more accurate approximation
    #we'll want to have a very small incremental update in order to get the instantaneous rate of change
    t_step = 0.01 
    while t < t_end:
        B_map = np.array([[-np.sin(y[1]), 0, 1], [np.sin(y[2])*np.cos(y[1]), np.cos(y[2]), 0], [np.cos(y[2])*np.cos(y[1]), -np.sin(y[2]), 0]])
        B_inv = np.linalg.inv(B_map)
        b_omega = np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])*(20*np.pi/180)
        #define the Euler angle differential kinematic equation which is the product of the inverse of the
        #coordinate transformations and the omega vector
        #the Euler angle differential kinematic equation gives us the instantaneous rates of change of our 
        #yaw-pitch-roll (3-2-1) Euler angles
        euler_dke = np.dot(B_inv, b_omega)
        #integration (dot multiplication of the kinematic differential equation with the respective 
        #small incremental timestep), giving us the updated attitude reached after the respective next timestep
        y_new = y + np.dot(euler_dke,t_step)
        y = y_new #updating the Euler attitude set
        t+=t_step #incrementing t
    return np.linalg.norm(y)

#print (euler_integrate(np.array([40*np.pi/180, 30*np.pi/180, 80*np.pi/180]), 42))

#Now, let's implement above integration algorithm for quaternions

def quaternion_integrate(beta0, t_end):
    beta = beta0 #define initial quaternion attitude description
    t = 0 #initial time is 0
    #make the time step as small as possible to get a more accurate approximation
    #we'll want to have a very small incremental update in order to get the instantaneous rate of change
    t_step = 0.01 
    while t < t_end:
        #below is the matrix-form of the differential beta (quaternion) components
        quat_delta = 0.5*np.array([[beta[0],-beta[1],-beta[2],-beta[3]],
        [beta[1],beta[0],-beta[3],beta[2]],
        [beta[2],beta[3],beta[0],-beta[1]],
        [beta[3],-beta[2],beta[1],beta[0]]])
        #as above, this is the given omega (angular velocity) vector
        b_omega = np.array([0,np.sin(0.1*t), 0.01, np.cos(0.1*t)])*(20*np.pi/180)
        #define the quaternion differential kinematic equation which is the product of the 
        #above matrix-form quaternion derivatives and the omega vector
        #the quaternion differential kinematic equation gives us the instantaneous rates of change of the 
        #four quaternion components, beta0, beta1, beta2 and beta3
        quaternion_dke = np.dot(quat_delta, b_omega)
        #integration (dot multiplication of the kinematic differential equation with the respective 
        #small incremental timestep), giving us the updated attitude reached after the respective next timestep
        beta_new = beta + np.dot(quaternion_dke,t_step)
        beta = beta_new #updating the quaternion attitude set
        t+=t_step #incrementing t
    return np.linalg.norm(beta[1:])

#print (quaternion_integrate(np.array([0.408248,0,0.408248,0.816497]), 42))

#Here is the implementation for Classical Rodrigues Parameters (CRP)

def crp_integrate(q0, t_end):
    q = q0 #define initial CRP attitude description
    t = 0 #initial time is 0
    #make the time step as small as possible to get a more accurate approximation
    #we'll want to have a very small incremental update in order to get the instantaneous rate of change
    t_step = 0.01 
    while t < t_end:
        #below is the matrix-form of the differential q (CRP) components
        crp_delta = 0.5*np.array([[1+np.square(q[0]), (q[0]*q[1])-q[2], (q[0]*q[2])+q[1]],
        [(q[1]*q[0])+q[2], 1+np.square(q[1]), (q[1]*q[2])-q[0]],
        [(q[2]*q[0])-q[1], (q[2]*q[1])+q[0], 1+np.square(q[2])]])
        #as above, this is the given omega (angular velocity) vector
        b_omega = np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])*(3*np.pi/180)
        #define the CRP differential kinematic equation which is the product of the 
        #above matrix-form quaternion derivatives and the omega vector
        #the CRP differential kinematic equation gives us the instantaneous rates of change of the 
        #three CRP components, q1, q2 and q3
        crp_dke = np.dot(crp_delta, b_omega)
        #integration (dot multiplication of the kinematic differential equation with the respective 
        #small incremental timestep), giving us the updated attitude reached after the respective next timestep
        q_new = q + np.dot(crp_dke,t_step)
        q = q_new #updating the CRP attitude set
        t+=t_step #incrementing t
    return np.linalg.norm(q),q

#Here is the implementation for Modified Rodrigues Parameters (MRP)

def mrp_integrate(sigma0, t_end):
    sigma = sigma0 #define initial MRP attitude description
    t = 0 #initial time is 0
    #make the time step as small as possible to get a more accurate approximation
    #we'll want to have a very small incremental update in order to get the instantaneous rate of change
    t_step = 0.01 
    while t < t_end:
        #below is the matrix-form of the differential sigma (MRP) components
        mrp_delta = 0.25*np.array([[1-np.dot(sigma, sigma)+2*np.square(sigma[0]), 2*((sigma[0]*sigma[1])-sigma[2]), 2*((sigma[0]*sigma[2])+sigma[1])],
        [2*((sigma[1]*sigma[0])+sigma[2]), 1-np.dot(sigma, sigma)+2*np.square(sigma[1]), 2*((sigma[1]*sigma[2])-sigma[0])],
        [2*((sigma[2]*sigma[0])-sigma[1]), 2*((sigma[2]*sigma[1])+sigma[0]), 1-np.dot(sigma, sigma)+2*np.square(sigma[2])]])
        #as above, this is the given omega (angular velocity) vector
        b_omega = np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])*(20*np.pi/180)
        #define the MRP differential kinematic equation which is the product of the 
        #above matrix-form quaternion derivatives and the omega vector
        #the MRP differential kinematic equation gives us the instantaneous rates of change of the 
        #three MRP components, sigma1, sigma2 and sigma3
        mrp_dke = np.dot(mrp_delta, b_omega)
        #integration (dot multiplication of the kinematic differential equation with the respective 
        #small incremental timestep), giving us the updated attitude reached after the respective next timestep
        sigma_new = sigma + np.dot(mrp_dke,t_step)
        sigma = sigma_new #updating the MRP attitude set
        t+=t_step #incrementing t
        #Now, the big difference with the MRP is that we get one of two possible attitude sets - one describing
        #a short (preferred) rotation, the other the long way around (usually not preferred). The module
        #'mrp_conversion.py' converts the respective long/short MRP sets.
        #We also want to avoid +/-360° singularities which we can do by below switching mechanism
        #For the integration, we'd want the updated attitude after a given period using short rotations.
        #So if our resulting sigma is a long rotation, we simply use above mentioned module to convert it into
        #the short rotation form (sigma's norm < 1 --> short rotation vs norm > 1 --> long rotation)
        #Note that this if statement could also be put at the top of the integration loop.
        if np.linalg.norm(sigma) > 1:
            sigma = convert_mrp(sigma)
    return np.linalg.norm(sigma)

#print (crp_integrate(np.array([0.4,0.2,-0.1]), 42))
print (mrp_integrate(np.array([0.4,0.2,-0.1]), 42))





