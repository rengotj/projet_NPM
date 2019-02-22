# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:31:26 2019
@author: juliette rengot

Compute neighboorhoods
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def brute_force_KNN(queries, supports, k):
    """ Select the k nearest neighbours """
    neighborhoods = []
    for query in queries :
        dist = np.linalg.norm(query-supports, axis=1)
        sample_idx = []
        for i in range(k) :
            sample_idx.append(np.argmin(dist))
            dist[sample_idx] = np.inf
        neighborhoods.append(sample_idx)

    return neighborhoods

def brute_force_spherical(queries, supports, radius):
    """ Select all the neighbours inside a sphere of given radius"""
    neighborhoods = []
    for query in queries :
        dist = np.linalg.norm(query-supports, axis=1)
        test = np.where(dist<radius)
        neighborhoods.append(supports[test[0]])
    
    return neighborhoods


def dot_product(matrix_1, matrix_2):
    assert(matrix_1.shape[0]==matrix_2.shape[0])
    result =[]
    for i in range(matrix_1.shape[0]):
        result.append(sum((a*b) for a, b in zip(matrix_1[i,:], matrix_2[i,:])))
    return np.array(result)

def compute_eimls(points, normals, volume, number_cells, min_grid, length_cell):
    #compute KDTree for nearest neighbours
    tree = KDTree(points)

    for i in range(number_cells+1):
        for j in range(number_cells+1):
            for k in range(number_cells+1): # look at every node of the grid
                node = np.array(min_grid + [i,j,k]*length_cell) #the considered node of the volume grid
                
                dist, idx = tree.query(np.array([node]), 10) #find the 10 nearest neighbours in the point cloud
                closest_point = points[idx[0][:]]
                hoppe = dot_product(normals[idx[0][:]], node-closest_point)
                
                h = [np.linalg.norm(node-closest_point[i,:])/4.0 for i in range(10)] #parameter ℎ will vary following the considered point
                h = np.clip(h, 0.003, np.max(h))

                theta = np.array([np.exp(-(np.linalg.norm(node-closest_point[i,:]))**2/h[i]**2) for i in range(10)])
                
                imls = np.sum(hoppe*theta)/np.sum(theta)
                volume[i,j,k] = imls
    return
                  
def k_ring(points, normals, k, plot=False):
	# Compute the min and max of the data points
    min_grid = np.copy(points[0, :])
    max_grid = np.copy(points[0, :])
    for i in range(1, points.shape[0]):
        for j in range(0, 3):
            if (points[i,j] < min_grid[j]):
                min_grid[j] = points[i,j]
            if (points[i,j] > max_grid[j]):
                max_grid[j] = points[i,j]
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)
	# Number_cells is the number of voxels in the grid in x, y, z axis
    number_cells = 30
    length_cell = np.array([(max_grid[0]-min_grid[0])/number_cells,(max_grid[1]-min_grid[1])/number_cells,(max_grid[2]-min_grid[2])/number_cells])
	# Create a volume grid to compute the scalar field for surface reconstruction
    volume_eimls = np.zeros((number_cells+1,number_cells+1,number_cells+1),dtype = np.float32)
	# Compute the scalar field in the grid
    compute_eimls(points, normals, volume_eimls, number_cells, min_grid, length_cell)
	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts_eimls, faces_eimls, normals_tri_eimls, values_tri_eimls = measure.marching_cubes_lewiner(volume_eimls, level=0.0, spacing=(length_cell[0],length_cell[1],length_cell[2]))

   # Plot the mesh
    if plot:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts_eimls[faces_eimls])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlim(0, number_cells*length_cell[0])
        ax.set_ylim(0, number_cells*length_cell[1])
        ax.set_zlim(0, number_cells*length_cell[2])
        ax.view_init(elev=90, azim=-90)
        plt.axis('off')
        plt.show()    
    
    #Find nearest neighbour
    neighborhood_direct = {}
    for f in faces_eimls :
        for v in range(3) :
            faces = list(f.copy())
            faces.pop(v)
            if f[v] in neighborhood_direct.keys():
                neighborhood_direct[f[v]] = list(np.unique(neighborhood_direct[f[v]]+faces))
            else:
                neighborhood_direct[f[v]] = faces
    print(len(neighborhood_direct.keys()))
    neighborhood = {}                              
    for i in range(1, k):
        for v in neighborhood_direct.keys():
            for neighbor in neighborhood_direct[v]:
                if v in neighborhood.keys():
                    neighborhood[v] = list(np.unique(neighborhood[v] + neighborhood_direct[neighbor]))
                else :
                    neighborhood[v] = list(np.unique(neighborhood_direct[v] + neighborhood_direct[neighbor]))
    
    return(neighborhood)