import os, sys
import numpy as np
import cv2 as cv
import copy

import scipy.io as io


def project_points(X, K, R, T, distortion_flag=False, distortion_params=None):
        """
        Your implementation goes here!
        """
        # Project points from 3d world coordinates to 2d image coordinates

        #get projection matrix
        pmatrix = projection_matrix(R, T, K)

        #add 4th component to points
        ones = np.ones([1,len(X[0])])
        xones=np.row_stack((X,ones))

        #calculate pixel coordinates
        X_camera = pmatrix.dot(xones)

        return X_camera



def distort(X, K, distortion_params):
    kinv = np.linalg.inv(K)
    xn=kinv.dot(X)

    for i in range(0,  np.shape(xn)[1]):
        xn0=xn[0,i]
        xn1=xn[1,i]
        r2=xn0**2 + xn1**2
        r4 = r2**2
        r6 = r2 * r4
        factor = (1 + distortion_params[0] * r2 + distortion_params[1] * r4 + distortion_params[4] * r6)
        xd0 = xn0 * factor
        xd1 = xn1 * factor
        xn[0,i]=xd0
        xn[1,i]=xd1
    x = K.dot(xn)
    return x

def distort3(X,K,distortion_params):
    kinv = np.linalg.inv(K)
    Xn = kinv.dot(X)
    for i in range(0, np.shape(X)[1]):

        xn=Xn[0,i]
        yn = Xn[1, i]

        r2= xn**2 + yn **2
        r4 = r2**2
        r6= r2 * r4
        factor = (1 + distortion_params[0] * r2 + distortion_params[1] * r4 + distortion_params[4] * r6)
        xd = xn * factor
        yd = yn * factor

        Xn[0,i] = xd
        Xn[1,i] = yd
    Xd= K.dot(Xn)
    print(Xd)
    return Xd


def projection_matrix(R, T, K):
    kmatrix = np.matrix(K)
    rt = np.column_stack((R,T))
    #print(R)

    #combine(stack) R and T Matrix, stack with (0,0,0,1)
    rtmatrix = np.matrix(np.row_stack((np.column_stack((R,T)),(0,0,0,1))))
    #print(rtmatrix)

    #multiply K matrix with (identiy stack zerovector)
    imatrix = np.identity(3)
    kio = np.matrix(kmatrix.dot(np.column_stack((imatrix,(0,0,0)))))
    #print(kio)

    #multiply all to get projection matrix P
    pmatrix = kio.dot(rtmatrix)
    #print(pmatrix)
    return pmatrix

def column(matrix, i):
    return [row[i] for row in matrix]


#returns column vector i of a matrix
def expoint(matrix, i):
    p = []
    for index in range(0, len(matrix)):
        p.append(matrix[index,i])
    return p

def project_and_draw(img, X_3d, K, R, T, distortion_flag, distortion_parameters):
    """
        Your implementation goes here!
    """
    # call your "project_points" function to project 3D points to camera coordinates
    # draw the projected points on the image and save your output image here
    # cv.imwrite(output_name, img_array)
    X_camera = project_points(X_3D,K,R,T,distortion_flag,distortion_parameters)

    newimg=copy.copy(img)
    color = (0, 230, 0)
    if not distortion_flag:
        color = (0,0,230)

    Xp = []
    Xp.append([])
    Xp.append([])

    for cur in range(0,np.shape(X_camera)[1]):
        x = X_camera[0,cur]
        y = X_camera[1,cur]
        z = X_camera[2,cur]
        xp = int(x/z)
        yp = int(y/z)
        Xp[0].append(xp)
        Xp[1].append(yp)
    Xp2 = np.row_stack((Xp,np.ones(len(Xp[0]))))
    if(distortion_flag):
        Xp2 = distort(Xp2,K,distortion_parameters)

    for cur in range(0, np.shape(X_camera)[1]):
        x = Xp2[0, cur]
        y = Xp2[1, cur]
        newimg = cv.circle(newimg, (int(x), int(y)), 2, color, 0)

    #cv.imshow("Test",newimg)
    #cv.waitKey(0)

    return newimg

if __name__ == '__main__':
    base_folder = './data/'

    image_num = 1
    data = io.loadmat('./data/ex1.mat')
    X_3D = data['X_3D'][0]
    TVecs = data['TVecs']		# Translation vector: as the world origin is seen from the camera coordinates
    RMats = data['RMats']		# Rotation matrices: converts coordinates from world to camera
    kc = data['dist_params']	# Distortion parameters
    Kintr = data['intinsic_matrix']	# K matrix of the cameras

    #print(project_points(X_3D,Kintr,RMats,TVecs))
    #print(projection_matrix(RMats[0],TVecs[0],Kintr))


    imgs = [cv.imread(base_folder+str(i).zfill(5)+'.jpg') for i in range(TVecs.shape[0])]

    for i in range(0,24):
        img = project_and_draw(imgs[image_num], X_3D, Kintr, RMats[image_num], TVecs[image_num],True, kc)
        cv.imwrite((str(i) + "_distCor.jpg"), img)
        img2 = project_and_draw(imgs[image_num], X_3D, Kintr, RMats[image_num], TVecs[image_num],False, kc)
        cv.imwrite((str(i) + "_nodistCor.jpg"), img2)

    #project_and_draw(imgs[image_num], X_3D, Kintr, RMats[image_num], TVecs[image_num], True, kc)

