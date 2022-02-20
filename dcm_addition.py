
#Python implementation of three steps of Euler angle manipulation
#1. Given two Euler angle sets, find the respective DCM representations
#2. Through DCM addition, find the final required attitude representation (consider two spacecrafts B and R
#   and an inertial frame N, we are given the orientation of B relative to N and of R relative to N, and now
#   we want to find B relative to R, i.e. first spacecraft relative to second spacecraft)
#3. Through inverse transformation, find the respective 3-2-1 Euler angles given the newly found DCM of BR

import numpy as np


def dcm_add(bna1, bna2, bna3, rna1, rna2, rna3):
    bna1 = bna1 * np.pi/180
    bna2 = bna2 * np.pi/180
    bna3 = bna3 * np.pi/180
    rna1 = rna1 * np.pi/180
    rna2 = rna2 * np.pi/180
    rna3 = rna3 * np.pi/180

    be1 = np.cos(bna2)*np.cos(bna1)
    be2 = np.cos(bna2)*np.sin(bna1)
    be3 = -np.sin(bna2)
    be4 = np.sin(bna3)*np.sin(bna2)*np.cos(bna1)-np.cos(bna3)*np.sin(bna1)
    be5 = np.sin(bna3)*np.sin(bna2)*np.sin(bna1)+np.cos(bna3)*np.cos(bna1)
    be6 = np.sin(bna3)*np.cos(bna2)
    be7 = np.cos(bna3)*np.sin(bna2)*np.cos(bna1)+np.sin(bna3)*np.sin(bna1)
    be8 = np.cos(bna3)*np.sin(bna2)*np.sin(bna1)-np.sin(bna3)*np.cos(bna1)
    be9 = np.cos(bna3)*np.cos(bna2)

    re1 = np.cos(rna2)*np.cos(rna1)
    re2 = np.cos(rna2)*np.sin(rna1)
    re3 = -np.sin(rna2)
    re4 = np.sin(rna3)*np.sin(rna2)*np.cos(rna1)-np.cos(rna3)*np.sin(rna1)
    re5 = np.sin(rna3)*np.sin(rna2)*np.sin(rna1)+np.cos(rna3)*np.cos(rna1)
    re6 = np.sin(rna3)*np.cos(rna2)
    re7 = np.cos(rna3)*np.sin(rna2)*np.cos(rna1)+np.sin(rna3)*np.sin(rna1)
    re8 = np.cos(rna3)*np.sin(rna2)*np.sin(rna1)-np.sin(rna3)*np.cos(rna1)
    re9 = np.cos(rna3)*np.cos(rna2)

    BN = np.array([[be1, be2, be3], [be4, be5, be6], [be7, be8, be9]])
    RN = np.array([[re1, re2, re3], [re4, re5, re6], [re7, re8, re9]])

    BR = np.matmul(BN, np.transpose(RN))

    ang1 = np.arctan2(BR[0][1], BR[0][0])
    ang2 = np.arcsin(BR[0][2])
    ang3 = np.arctan2(BR[1][2], BR[2][2])
    return BN, np.rad2deg(ang1), np.rad2deg(ang2), np.rad2deg(ang3)

print (dcm_add(20,-10,120,-5,5,5))













