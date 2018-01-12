import os, sys
import numpy as np
import cv2 as cv
import copy
import scipy.io as io
import random
import matplotlib


def compute_F(K_0,K_1,R,t):
    Kit_1 = np.linalg.inv(np.transpose(K_1))
    tx = createaxMatrix(t)
    Ki_0 = np.linalg.inv(K_0)
    F=np.linalg.multi_dot([Kit_1,tx,R,Ki_0])
    return F


def compute_epipolarLine(c,F):
    lp = np.dot(F,c)
    return lp

def compute_epipolarLines(c0,F,c1):
    L = []
    corresponding = []
    for c in c0:
        c3d = create3dfrom2dpoint(c)
        Fc = compute_epipolarLine(c3d, F)
        cmin = findMinimumC3(c, c1, F)
        corresponding.append(cmin)
        L.append([Fc[0,0],Fc[0,1],Fc[0,2]])
    corresponding = np.matrix(corresponding)
    L= np.matrix(L)
    return (L,corresponding)

def drawEpipolarLines(newimg,L,points):
    for i in range(0,len(L)-1):
        a = L[i,0]
        b=L[i,1]
        c=L[i,2]
        x0,y0=map(int,[0,-c/b])
        x1,y1 = map(int,[points[i,0],-(c+(a*points[i,0]))/b])
        color = (random.randint(0,250),random.randint(0,250),random.randint(0,250))
        newimg = cv.line(newimg,(x0,y0),(x1,y1),color,3)
        newimg = cv.circle(newimg, (int(points[i,0]),int(points[i,1])), 5, color, 5)
    return newimg

def findMinimumC3(c,c1,F):
    min = 100
    newC = None
    for cmin in c1:
        curmin = np.abs(np.linalg.multi_dot([np.transpose(create3dfrom2dpoint(cmin)),F,create3dfrom2dpoint(c)]))
        if curmin < min:
            min = curmin
            newC=cmin
    return newC

def createStackImage(img0,img1,c0,p1):
    yoffset = 3168
    stack = np.concatenate((img0, img1), axis=0)
    for i in range(0,len(c0)-1):
        color = (random.randint(0,250),random.randint(0,250),random.randint(0,250))
        stack = cv.circle(stack, (int(c0[i,0]),int(c0[i,1])), 10, color, 10)
        stack = cv.circle(stack, (int(p1[i,0]),int(p1[i,1]+yoffset)), 10, color, 10)
        stack = cv.line(stack,(int(p1[i,0]),int(p1[i,1]+yoffset)),(int(c0[i,0]),int(c0[i,1])),color,5)
    return stack

def createaxMatrix(a):
    ax=0
    if len(a[0])==3:
        ax = np.matrix([[0,-a[0,2],a[0,1]],[a[0,2],0,-a[0,0]],[-a[0,1],a[0,0],0]])
    else:
        print("Not a compatible vector")
        ax = np.identity(3)
    return ax

def create3dfrom2dpoint(v2d):
    if len(v2d)==2:
        v3d = [v2d[0],v2d[1],1]
    else:
        print("wrong vector dimension")
        v3d = [1,1,1]
    return v3d

def mapFeatures(K_0,K_1,R,t,c0,c1,img0,img1):
    newimg = copy.copy(img1)
    
    F = compute_F(K_0,K_1,R,t)
    print("Fundamental Matrix is : ",F)

    L,cminpoint = compute_epipolarLines(c0,F,c1)
    newimg = drawEpipolarLines(newimg,L,cminpoint)

    cv.imwrite("epilines.jpg",newimg)
    stackImage = createStackImage(img0,img1,c0,cminpoint)
    cv.imwrite("matches.jpg",stackImage)
    return matchingFeatures

if __name__ == '__main__':
    base_folder = './data/'
    dict=io.loadmat(base_folder + 'data.mat')
    K_0 =dict["K_0"]
    K_1 =dict["K_1"]
    R_1 =dict["R_1"]
    t_1 =dict["t_1"]
    cornersCam0 =dict["cornersCam0"]
    cornersCam1 =dict["cornersCam1"]
    img0 = cv.imread(base_folder + 'Camera00.jpg')
    img1 = cv.imread(base_folder + 'Camera01.jpg')

    matchingFeatures = mapFeatures(K_0,K_1,R_1,t_1,cornersCam0,cornersCam1,img0,img1)




