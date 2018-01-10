import os, sys
import numpy as np
import cv2 as cv
import copy
import scipy.io as io
import random


def compute_F(K_0,K_1,R,t):
    Kit_1 = np.transpose(np.linalg.inv(K_1))
    tx = createaxMatrix(t)
    Ki_0 = np.linalg.inv(K_0)
    F=np.linalg.multi_dot([Kit_1,tx,R,Ki_0])
    return F


def compute_epipolarLine(c,F):
    lp = np.dot(F,c)
    return lp

def compute_epipolarLines(c0,F):
    L = []
    for c in c0:
        c3d = create3dfrom2dpoint(c)
        Fc = compute_epipolarLine(c3d, F)
        #print(Fc)
        L.append([Fc[0,0],Fc[0,1],Fc[0,2]])
    L= np.matrix(L)
    return L

def drawEpipolarLines(newimg,L):
    row = 3168
    col = 4752
    color = (0, 0, 230)

    for l in L:
        a = l[0,0]
        b=l[0,1]
        c=l[0,2]
        fact0 = -10
        fact1= 10
        acb = -((a+c)/b)


        x0,y0=map(int,[0,-c/b])
        x1,y1=map(int,[col,-(c+a*col)/b])
        #print(x0,y0,x1,y1)
        newimg = cv.line(newimg,(x0,y0),(x1,y1),color,3)
    return newimg


def computeMatchingFeature(c1,L):
    p0 =[]
    p1=[]
    #print(L)
    #print(c1)
    for l in L:
        c = findMinimumC2(c1,l)
        p1.append(c)
    p1= np.matrix(p1)
    #print(p1)
    return p1

def findMinimumC(c1,l):
    a = l[0, 0]
    b = l[0, 1]
    c = l[0, 2]
    p = [a, b]
    n = [1, (a + c *4752/ -b)]
    n = n / np.linalg.norm(n)
    if np.dot(p, n) < 0:
        n = -n
    d = np.dot(p, n)


    minD = np.abs(np.dot(c1[0], n) - d)
    minC = c1[0]
    for c in c1:
        curD=np.abs(np.dot(c, n) - d)
        if curD < minD:
            minD = curD
            minC = c
    return minC

def findMinimumC2(c1,l):
    row = 3168
    col = 4752
    a = l[0, 0]
    b = l[0, 1]
    c = l[0, 2]
    x0, y0 = map(int, [0, -c / b])
    x1, y1 = map(int, [col, -(c + a * col) / b])

    minD = dist((x0, y0), (x1, y1), c1[0])
    minC = c1[0]
    for c in c1:
        curD = dist((x0,y0),(x1,y1),c)
        if curD < minD:
            minD = curD
            minC = c
    return minC




def dist((x1,y1),(x2,y2),(x0,y0)):
    d1 = np.abs((y2 -y1)*x0 - (x2-x1) *y0 + x2 * y1 - y2*x1)
    d2 = np.sqrt((y2-y1)**2+(x2-x1)**2)
    dist = d1/d2
    return dist
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
    color = (0, 0, 230)
    xoffset = 2376
    yoffset = 1584


    F = compute_F(K_0,K_1,R,t)

    L = compute_epipolarLines(c0,F)



    newimg = drawEpipolarLines(newimg,L)

    cv.imwrite("epilines.jpg",newimg)
    #cv.imshow("1",newimg)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #print(c1)
    p1 = computeMatchingFeature(c1,L)
    #print(len(c0),len(p1))
    #print(p1)

    #print(c0[5])
    #print(p1)
    #print(p1[0,0])
    stackImage = createStackImage(img0,img1,c0,p1)
    cv.imwrite("matches.jpg",stackImage)


def reconstructStructure():
    y=0


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
    #print(create3dfrom2dpoint(cornersCam0[0]))

    mapFeatures(K_1,K_0,R_1,t_1,cornersCam0,cornersCam1,img0,img1)



