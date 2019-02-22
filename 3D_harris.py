# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:27:42 2019
@author: juliette rengot

A robust 3D IPD based on harris detector
"""
import itertools
import scipy
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from math import cos, sin, pi, sqrt

import sys
sys.path.append("C://Users//juliette//Desktop//enpc//3A//S2//Nuage_de_points_et_modélisation_3D//projet//github//utils")
from ply import write_ply, read_ply
#from utils.ply import write_ply, read_ply

#import warnings
#warnings.filterwarnings("ignore")

import neighborhoords
import transformation



def dot_product(vector_1, vector_2):
  return sum((a*b) for a, b in zip(vector_1, vector_2))
def length(vector):
  return sqrt(dot_product(vector, vector))
def angle(v1, v2):
    a = dot_product(v1, v2)/(length(v1)*length(v2))
    a = np.clip(a, -1, 1)
    return(np.arccos(a))

def polyfit3d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

if __name__ == '__main__':
    #parameters
    plot = False
    n_neighbours = 3
    k = 0.04
    fraction = 0.1
    cluster_threshold = 0.01
    scale_range = [0.5, 2.0]
    offset_range = [10**(-4), 10**(-3)]
    
    # Load point cloud
    file_path = 'C://Users//juliette//Desktop//enpc//3A//S2//Nuage_de_points_et_modélisation_3D//projet//github//data//bunny_normals.ply' # Path of the file
    data = read_ply(file_path)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
                       
    #initialisation of the solution
    labels_fraction = np.zeros(len(points))
    labels_cluster = np.zeros(len(points))
    resp = np.zeros(len(points))
    
    #compute neighborhood
    #TODO : use k ring (vert == points)
#    neighborhood = neighborhoords.k_ring(points, normals, n_neighbours, plot)
    neighborhood = neighborhoords.brute_force_KNN(points, points, n_neighbours)
    
    for i in range(len(points)) :
        neighbors = points[neighborhood[i], :]
        points_centred, _ = transformation.centering_centroid(points)
        
        #best fitting point
        pca = PCA(n_components=3) #Principal Component Analysis
        points_pca = pca.fit_transform(np.transpose(points_centred))
        eigenvalues, eigenvectors = np.linalg.eigh(points_pca)
        idx = np.argmin(eigenvalues, axis=0)
        best_fit_normal = eigenvectors[idx,:]
        
        #rotate the cloud
        #TODO : corriger rotation
#        theta = angle(best_fit_normal, np.array([1,0,0])) #angle with x axis
#        points_rotated = transformation.rotation(points, theta, 2)
#        theta = angle(best_fit_normal, np.array([0, 0, 1])) #angle with z axis
#        points_rotated = transformation.rotation(points_rotated, theta, 1)
        for i in range(points.shape[0]):
            points[i, :] = np.dot(np.transpose(eigenvectors), points[i, :])
            
        #restrict to XY plane and translate
        points_2D = points[:,:2]-points[i,:2]
        
        #fit a quadratic surface
        #TODO : other method / correct poly form
        m = polyfit3d(points_2D[:,0], points_2D[:,1], points[:,2], order=2)
        m = m.reshape((3,3))
#        p = np.polyfit(points_2D[:,0], points_2D[:,1], 2) #Least squares polynomial fit.
#        p = np.poly1d(p)
#        print(p)
        
        #Compute the derivative
        #TODO : ajouter integration gaussienne
        fx2 = m[1, 0]*m[1, 0]  + 2*m[2, 0]*m[2, 0] + 2*m[1, 1]*m[1, 1] #A
        fy2 = m[1, 0]*m[1, 0]  + 2*m[1, 1]*m[1, 1] + 2*m[0, 2]*m[0, 2] #B
        fxfy = m[1, 0]*m[0, 1] + 2*m[2, 0]*m[1, 1] + 2*m[1, 1]*m[0, 2] #C
          
        #Compute response
        resp[i] = fx2*fy2 - fxfy*fxfy - k*(fx2 + fy2)*(fx2 + fy2);
                                       
        #Select interest points
        #Todo : NMS / ANMS

    #Select interest points
    #search for local maxima
    candidate = []
    for i in range(len(points)) :
        if resp[i] >= np.max(resp[neighborhood[i]]) :
            candidate.append([i, resp[i]])
    #sort by decreasing order
    candidate.sort(reverse=True, key=lambda x:x[1])
    candidate = np.array(candidate)
    
    #Method 1 : fraction
    interest_points = np.array(candidate[:int(fraction*len(points)), 0], dtype=np.int)
    labels_fraction[interest_points] = 1
    
    #Method 2 : cluster
    Q = points[int(candidate[0, 0]), :].reshape((1,-1))
    for i in range(1, len(candidate)) :
        query = points[int(candidate[i, 0]), :].reshape((1,-1))
        distances = scipy.spatial.distance.cdist(query, Q, metric='euclidean')
        if np.min(distances) > cluster_threshold :
            Q = np.concatenate((Q, query), axis=0)
            labels_cluster[int(candidate[i, 0])] = 1
          
    # Save the result
    write_ply('../bunny_harris_IPD.ply', [points, labels_fraction, labels_cluster], ['x', 'y', 'z', 'labels_fraction', 'labels_cluster'])
    