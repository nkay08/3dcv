import os, sys
import numpy as np
import cv2 as cv
import copy
import scipy.io as io


def compute_F(K_0,K_1,R,t):
    K_1nt = np.transpose(np.linalg.inv(K_1))
    tx = createaxMatrix(t)
    K_0n = np.transpose(K_0)
    F=np.linalg.multi_dot([K_1nt,tx,R,K_0n])
    return F


def compute_epipolarLine(c,F):
    y=0

def drawEpipolarLine():
    y=0

def computeMatchingFeature():
    y=0

def createStackImage():
    y=0

def createaxMatrix(a):
    ax=0
    if len(a[0])==3:
        ax = np.matrix([[0,-a[0,2],a[0,1]],[a[0,2],0,-a[0,0]],[-a[0,1],a[0,0],0]])
    else:
        print("Not a compatible vector")
    return ax


def mapFeatures(K_0,K_1,r,t,c0,c1):
    y=0


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



