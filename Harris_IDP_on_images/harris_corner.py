# -*- coding: utf-8 -*-
"""
@author: juliette rengot

Harris Interest point dection for image processing
"""

import cv2
import numpy as np
import os
from math import sqrt
import matplotlib.pyplot as plt


k_NMS = 0       #if k_NMS=0 : don't use Non-maximum suppression
                #if 0<k_NMS<1 : use non-masimum suppression with almost largest condition
                #if k_NMS>=1 : use Non-maximum suppression on (2*k_NMS+1)*(2*k_NMS+1) neighbours

N_ANMS = 1000      # if N_ANMS=0 : don't use Adaptative Non-maximum suppression
                # if N_ANMS>0 : don't use Adaptative Non-maximum suppression to obtain N_ANMS corners
c_ANMS = 0.9    # criterion for ANMS (used if N_ANMS>0)

#can't use both NMS and ANMS
assert(k_NMS!=0 or N_ANMS!=0)

repeat_test = False  #Test the repeatability of the experience
deform = False       #Try to apply a deformation to an image and see the impact on the detection


def NMS(cornerList,responseMatrix, k):
    """ Non Maximum suppression """
    w=responseMatrix.shape[0]
    h=responseMatrix.shape[1]
    
    newCornerList = []
    for corner in cornerList :
        if(0<=corner[0]-k and 0<=corner[1]-k and corner[0]+k+1<w and corner[1]+k+1<h):
            subResponseMatrix = responseMatrix[corner[0]-k:corner[0]+k+1, corner[1]-k:corner[1]+k+1]
            if responseMatrix[corner[0],corner[1]] == subResponseMatrix.max():
                newCornerList.append(corner)
    
    return(newCornerList)


def almost_largest(cornerList, responseMatrix, c):
    """ Non Maximum suppression with almost largest criterion"""
    w=responseMatrix.shape[0]
    h=responseMatrix.shape[1]
    
    newCornerList = []
    for corner in cornerList :
        test = True
        for x in range(max(0,corner[0]-1), min(corner[0]+2,w-1)):
            for y in range(max(0,corner[1]-1),min(corner[1]+2,h-1)):
                r = responseMatrix[x,y]
                if c*r > responseMatrix[corner[0],corner[1]] :
                    test = False
        if(test):
            newCornerList.append(corner)
    return(newCornerList)


def find_idx(responseList,r):
    """ Find where to place r in responseList to make responseList sorted in reverse order"""
    if len(responseList)==0 :
        return(0)
    idx = 0
    while responseList[idx]>=r:
        idx+=1
        if idx==len(responseList) :
            return(-1)
    return(idx)


def order_resp(cornerList, responseMatrix):
    """ Order cornerList to first have the corner with higher corner response"""
    newCornerList = []
    responseList = []
    for corner in cornerList:
        r = responseMatrix[corner[0],corner[1]]
        idx = find_idx(responseList, r)
        responseList.insert(idx,r)
        newCornerList.insert(idx,corner)
    return(newCornerList)

def order_ray(cornerList, rayList):
    """ Order cornerList to first have the corner with higher radius"""
    newCornerList = []
    newRayList = []
    for i in range(len(cornerList)):
        corner=cornerList[i]
        r = rayList[i]
        idx = find_idx(newRayList, r)
        newRayList.insert(idx, r)
        newCornerList.insert(idx, corner)
    return(newCornerList)
        
def ANMS(cornerList, responseMatrix, c, N):
    """ Adaptive Non Maximum suppression """
    #Sort : Descending order of corner response
    cornerList = order_resp(cornerList, responseMatrix)
    #Initialisation
    ray_INF = 100000
    cornerProcessedList = [cornerList[0]]
    rayList = [ray_INF]
    #Computes radius for each corner
    for corner in cornerList[1:] :
        ray_min = ray_INF
        for corner_processed in cornerProcessedList :
            if c*responseMatrix[corner_processed[0],corner_processed[1]] <= responseMatrix[corner[0],corner[1]] :
                ray = np.linalg.norm(np.array(corner)-np.array(corner_processed))
                ray_min = min(ray,ray_min)
        cornerProcessedList.append(corner)
        rayList.append(ray_min) 
    #sort by radius
    finalCornerList=order_ray(cornerProcessedList, rayList)
    return(finalCornerList[0:N])


def findCorners(img, k_NMS):
    """ Harris corner detection """
    #Parameters
    thresh=100000
    k=0.2
    offset=2
    
    #Find derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    
    h = img.shape[0]
    w = img.shape[1]
    cornerList = []
    responseMatrix = np.zeros(img.shape)
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)

    #Loop through image and find our corners
    for x in range(offset, h-offset):
        for y in range(offset, w-offset):
            windowIxx = Ixx[x-offset:x+offset+1, y-offset:y+offset+1]          
            windowIxy = Ixy[x-offset:x+offset+1, y-offset:y+offset+1]
            windowIyy = Iyy[x-offset:x+offset+1, y-offset:y+offset+1]
            
            #smoothing
            kernel = 1/sqrt(273)*np.array([1, 4, 7, 4, 1])
            windowIxx = np.dot(np.transpose(kernel), np.dot(kernel,windowIxx))
            windowIxy = np.dot(np.transpose(kernel), np.dot(kernel,windowIxy))
            windowIyy = np.dot(np.transpose(kernel), np.dot(kernel,windowIyy))
            
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy**2) # determinant
            trace = Sxx + Syy            # trace
            r = det - k*(trace**2)       # corner response

            if (r > thresh):             # Corner Detect 
                responseMatrix[x,y] = r
                cornerList.append([x, y])
                
                if (k_NMS == 0 and N_ANMS == 0):
                    # Visualisation
                    color_img.itemset((x, y, 0), 0)
                    color_img.itemset((x, y, 1), 0)
                    color_img.itemset((x, y, 2), 255)
                    newCornerList = cornerList

    if (k_NMS >= 1):
        print("Non local Maximum suppression")
        newCornerList = NMS(cornerList, responseMatrix, k_NMS)
        # Visualisation
        for corner in newCornerList:
            color_img.itemset((corner[0], corner[1], 0), 0)
            color_img.itemset((corner[0], corner[1], 1), 0)
            color_img.itemset((corner[0], corner[1], 2), 255)
            
    elif (0<k_NMS<1):
        print("Non local Maximum suppression with almost largest condition")
        newCornerList = almost_largest(cornerList, responseMatrix, k_NMS)
        # Visualisation
        for corner in newCornerList:
            color_img.itemset((corner[0], corner[1], 0), 0)
            color_img.itemset((corner[0], corner[1], 1), 0)
            color_img.itemset((corner[0], corner[1], 2), 255)
    
    elif(N_ANMS>0):
        print("Adaptive Non local Maximum suppression")
        newCornerList = ANMS(cornerList, responseMatrix, c_ANMS, N_ANMS)
        # Visualisation
        for corner in newCornerList:
            color_img.itemset((corner[0], corner[1], 0), 0)
            color_img.itemset((corner[0], corner[1], 1), 0)
            color_img.itemset((corner[0], corner[1], 2), 255)
            
    return(color_img, newCornerList)


def image_treatment(img_name, k_NMS, plot=True):
    """ Apply harris corner detection to an image """
    img = cv2.imread(img_name, 0)
    if img is None:
        print("Not able to open the image ", img)
    
    if (plot) :
        cv2.imshow('Original Image', img)
        print("Print original image : press key...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cornerImg, cornerList = findCorners(img, k_NMS)
    print("We detect ",len(cornerList)," corners.")

    if ((cornerImg is not None) and (plot)):
        cv2.imshow('Corner detection', cornerImg)
        print("Print corner detection : press key...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return(cornerList)


if __name__ == "__main__":   
    if (repeat_test):
        diff_list=[]
        corners_ref = image_treatment('images/notreDame2.jpg', k_NMS, False)
        for i in range(100):
            corners = image_treatment('images/notreDame2.jpg', k_NMS, False)
            diff=0
            for i in range(len(corners_ref)):
                if corners_ref[i] not in corners:
                    diff+=1
            diff_list.append(diff)
        plt.figure(1)
        plt.scatter([i for i in range(len(diff_list))], diff_list)
        plt.title('Repeatability test')
        plt.xlabel("experience number")
        plt.ylabel("Number of missing corner")
        plt.show()
        
    elif(deform):
        corner_ref = image_treatment("images/squares.png", k_NMS)
        
        H = np.array([[2,-1],[-1,2]])
        img = cv2.imread("images/squares.png", 0)
        img2 = np.zeros(img.shape)
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                coord=np.dot(H,np.array([x,y]))
                if 0<coord[0]<img.shape[0] and 0<coord[1]<img.shape[1]:
                    img2[x,y]=img[coord[0],coord[1]]
        cv2.imwrite("images/squares_deformed.png", img2)
        corners = image_treatment("images/squares_deformed.png", k_NMS)
        
    else:
        for file in os.listdir('images'):
            print("New Image : ", file)
            image_treatment('images/' + file, k_NMS)
            