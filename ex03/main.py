import os, sys
import numpy as np
import cv2 as cv
import copy

import scipy.io as io

def compute_relative_rotation(h,k):

    # H = lambda K [r1 r2 t]

    kinv = np.invert(k)

    kinvh = np.dot(kinv,h)
    print(kinvh)

    r1 = kinvh[:,0]
    r2 = kinvh[:,1]
    #r1= column(kinvh,0)
    #r2 = column(kinvh, 1)
    #t = h[:,2]

    #print(r1)
    #print(r2)
    norm1 = np.linalg.norm(r1)
    norm2 =np.linalg.norm(r2)
    #print(norm1,norm2)

    norm = (norm1+norm2)/2
    #print(norm)

    r1= r1/norm
    r2= r2/norm

    r3 = np.transpose(np.cross(np.transpose(r1),np.transpose(r2)))

    #print(r3)

    r = np.column_stack((r1,r2))
    r = np.column_stack((r,r3))

    print("Rotation Matrix is: ")
    print(r)

    r = checkRotationMatrix(r)
    #print(r)


def column(matrix, i):
    return [row[i] for row in matrix]

def checkRotationMatrix(r):
    r2= r

    rt = np.transpose(r)
    identity = np.identity(3)

    res = np.dot(r,rt)
    print(res)
    print(np.linalg.det(r))

    if not np.allclose(res,identity):
        print("Matrix needs correction")
        r2 = correctRotationMatrix(r)
        print("corrected Matrix is: ")
        print(r2)


    return r2


def correctRotationMatrix(r):
    U,W,V = np.linalg.svd(r)
    #print("u: ")
    #print(U)
    #print(W)
    #print("v: ")
    #print(V)
    Vt = np.transpose(V)
    r2= np.dot(U,Vt)
    #print(r2)
    return(r2)




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
    #compute_relative_rotation(h3,K)