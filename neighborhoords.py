# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:31:26 2019
@author: juliette rengot

Compute neighboorhoods
"""

import numpy as np
from scipy.spatial import Delaunay


def brute_force_KNN(points, k):
    """ Select the k nearest neighbours """
    neighborhoods = []
    for query in points :
        dist = np.linalg.norm(query-points, axis=1)
        sample_idx = []
        for i in range(k) :
            sample_idx.append(np.argmin(dist))
            dist[sample_idx] = np.inf
        neighborhoods.append(sample_idx)
    return neighborhoods


def brute_force_spherical(points, radius):
    """ Select all the neighbours inside a sphere of given radius"""
    neighborhoods = []
    for query in points :
        dist = np.linalg.norm(query-points, axis=1)
        test = np.where(dist<radius)
        neighborhoods.append(test[0])
    return neighborhoods


def k_ring_delaunay(points, k):
    """ Select all the neighbours in the k first rings"""
    triangulation = Delaunay(points) # Compute structure

    neighborhood_direct = {}
    for f in triangulation.simplices :
        for v in range(f.shape[0]) :
            faces = list(f.copy())
            faces.pop(v)
            if f[v] in neighborhood_direct.keys():
                neighborhood_direct[f[v]] = list(np.unique(neighborhood_direct[f[v]]+faces))
            else:
                neighborhood_direct[f[v]] = faces

    neighborhood = {}                              
    for i in range(1, k):
        for v in neighborhood_direct.keys():
            for neighbor in neighborhood_direct[v]:
                if v in neighborhood.keys():
                    neighborhood[v] = list(np.unique(neighborhood[v] + neighborhood_direct[neighbor]))
                else :
                    neighborhood[v] = list(np.unique(neighborhood_direct[v] + neighborhood_direct[neighbor]))
    
    return(neighborhood)