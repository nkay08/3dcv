import os, sys
import numpy as np
import cv2 as cv
import copy

import scipy.io as io

def compute_relative_rotation(H,K):
    # page 13: t is assumed 0
    # KRK^-1 x' =  H x'
    # R = K^-1 H K

    #Kinv = np.invert(K)
    Kinv = np.linalg.inv(K)

    KHK = np.linalg.multi_dot([Kinv,H,K])

    R= KHK

    print("Tentative Rrel is:")
    print(R)

    det = np.linalg.det(R)

    print("Determinant is:")
    print(det)


    if not np.isclose(det, 1,atol=0.009) and not np.isclose(det, -1,atol=0.009):
        print("Matrix needs correction")
        U, S, VT = np.linalg.svd(R)
        RRel = np.dot(U, VT)

        R = RRel
        print("New Rrel is:")
        print(R)

    return R

def compute_pose(H,K):
    #same as compute_rotation() ??
    R = None
    t = None


    # H = lambda K [r1 r2 t]

    #Kinv = np.invert(K)
    Kinv = np.linalg.inv(K)

    #matrix A = K-1 * H
    M = np.dot(Kinv,H)
    #print(A)

    m1 = M[:,0]
    m2 = M[:,1]
    m3 = M[:,2]
    lambda1 = np.linalg.norm(m1)
    lambda2 = np.linalg.norm(m2)
    #print(norm1,norm2)
    lambda0 = (lambda1+lambda2)/2
    #print(norm)

    r1=m1/lambda0
    r2=m2/lambda0

    print(r1)
    print(r2)

    r3 = np.transpose(np.cross(np.transpose(r1),np.transpose(r2)))

    #print(r3)

    r = np.column_stack((r1,r2))
    r = np.column_stack((r,r3))


    print("Rotation Matrix is: ")
    print(r)
    #R = checkRotationMatrix(r)
    R=  r
    t= m3/lambda0

    print("Translation is:")
    print(t)

    return R,t

if __name__ == '__main__':
    base_folder = './data/'
    data = io.loadmat('./data/ex2.mat')
    #print(data)
    alpha_x= data['alpha_x']
    alpha_y = data['alpha_y']
    x_0=data['x_0']
    y_0 = data['y_0']
    s= data['s']
    h1 = data['H1']
    h2 = data['H2']
    h3 = data['H3']


    K= np.matrix([[alpha_x[0,0],s[0,0],x_0[0,0]],[0,alpha_y[0,0],y_0[0,0]],[0,0,1]])

    compute_relative_rotation(h1,K)
    compute_relative_rotation(h2,K)

    compute_pose(h3,K)