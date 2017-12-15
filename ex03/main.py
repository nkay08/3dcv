import os, sys
import numpy as np
import cv2 as cv
import copy

import scipy.io as io

def compute_relative_rotation(H,K):

    # H = lambda K [r1 r2 t]

    Kinv = np.invert(K)

    #matrix A = K-1 * H
    A = np.dot(Kinv,H)
    #print(A)

    a1 = A[:,0]
    a2 = A[:,1]

    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    print(norm1,norm2)
    norm = (norm1+norm2)/2
    #print(norm)

    r1=a1/norm1
    r2=a2/norm2

    #r1= r1/norm
    #r2= r2/norm


    #print(r1)
    #print(r2)

    a3 = np.transpose(np.cross(np.transpose(r1),np.transpose(r2)))
    r3= a3/np.linalg.norm(a3)

    #print(r3)

    r = np.column_stack((r1,r2))
    r = np.column_stack((r,r3))


    print("Rotation Matrix is: ")
    print(r)



    r = checkRotationMatrix(r)

    #print(r)

    #result calculated by OpenCV
    #cvr = None
    #cvt=None
    #cvr = cv.decomposeHomographyMat(H,K,cvr,cvt)


    #print("CV: ")
    #print(cvr)


def compute_relative_rotation2(H,K):
    # page 13: t is assumed 0
    # KRK^-1 x' =  H x'
    # R = K^-1 H K
    # => R_i = K H_i K^-1 ?

    Kinv = np.invert(K)

    KHK = np.dot(np.dot(Kinv,H),K)


    #a1 = np.dot(np.dot(Kinv,H[:,0]),K)
    #a2 = np.dot(np.dot(Kinv, H[:, 1]), K)
    #a3 = np.dot(np.dot(Kinv, H[:, 2]), K)
    a1 = KHK[:,0]
    a2 = KHK[:,1]
    a3 = KHK[:,2]
    print(KHK)

    r1= a1/np.linalg.norm(a1)
    r2= a2/np.linalg.norm(a2)
    r3 = a3/np.linalg.norm(a3)

    #r3 = np.cross(r1,r2)

    #print(r1,r2,r3)
    #R = np.matrix(np.transpose(r1))
    #R = np.column_stack((R,np.transpose(r2)))
    #R = np.column_stack((R, np.transpose(r3)))

    R = np.matrix(a1)
    R = np.column_stack((R,a2))
    R = np.column_stack((R, a3))

    print("Rreel is:")
    print(R)

    det = np.linalg.det(R)

    print("Determinant is:")
    print(det)

    if not np.isclose(det,1) or np.isclose(det,-1):
        print("Matrix needs correction")
        U,S,V = np.linalg.svd(R)
        RRel = np.dot(U,V)

        R = RRel
        print("New Rrel is:")
        print(R)

    return R

def column(matrix, i):
    return [row[i] for row in matrix]

def checkRotationMatrix(r):
    r2= r

    rt = np.transpose(r)
    identity = np.identity(3)

    res = np.dot(r,rt)
    #print(res)
    print("Determinant must be close to 1 and is:")
    print(np.linalg.det(r))

    if not np.allclose(res,identity):
        print("Matrix needs correction")
        r2 = correctRotationMatrix(r)
        print("corrected rotation Matrix is: ")
        print(r2)
        print("New determinant is: ")
        print(np.linalg.det(r2))


    return r2


def correctRotationMatrix(r):
    U,W,V = np.linalg.svd(r)
    #print("u: ")
    #print(U)
    #print(W)
    #print("v: ")
    #print(V)
    Vt = np.transpose(V)
    uvt= np.dot(U,Vt)

    #r2= np.dot(uvt,r)
    r2= uvt
    #print(r2)
    return(r2)


def compute_pose(H,K):
    #same as compute_rotation() ??
    R = None
    t = None


    # H = lambda K [r1 r2 t]

    Kinv = np.invert(K)

    #matrix A = K-1 * H
    A = np.dot(Kinv,H)
    #print(A)

    a1 = A[:,0]
    a2 = A[:,1]

    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    #print(norm1,norm2)
    norm = (norm1+norm2)/2
    #print(norm)

    r1=a1/norm1
    r2=a2/norm2
    #r1= r1/norm
    #r2= r2/norm

    #print(r1)
    #print(r2)

    a3 = np.transpose(np.cross(np.transpose(r1),np.transpose(r2)))
    r3= a3/np.linalg.norm(a3)

    #print(r3)

    r = np.column_stack((r1,r2))
    r = np.column_stack((r,r3))


    print("Rotation Matrix is: ")
    print(r)
    R = checkRotationMatrix(r)

    t= A[:,2]/norm

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


    #compute_relative_rotation(h1,K)
    #compute_relative_rotation(h2,K)
    #compute_relative_rotation2(h1,K)
    #compute_relative_rotation2(h2,K)

    #compute_pose(h3,K)