import numpy as np
import math

import sys
sys.path.append("C://Users//juliette//Desktop//enpc//3A//S2//Nuage_de_points_et_modélisation_3D//projet//github//utils")
from ply import write_ply, read_ply
#from utils.ply import write_ply, read_ply


def centering_centroid(points):
    """center the point cloud on its centroid"""
    centred_points = points.copy()
    centroid = np.mean(centred_points, axis=0)
    centred_points = centred_points - centroid
    return(centred_points, centroid)

def scaling(points, factor):
    """rescale the point cloud"""
    scaled_points = points.copy()
    scaled_points = scaled_points/factor
    return(scaled_points)

def rotation(points, angle, axis):
    """Apply a rotation of a given angle
    around a given axis
    to a given point cloud"""
    assert(axis in [0,1,2])
    rotated_points = points.copy()
    if axis == 0 : # x axis
        rot_mat = np.array([[1, 0, 0],
                            [0, math.cos(angle), -math.sin(angle)],
                            [0, math.sin(angle), math.cos(angle)]])
    elif axis == 1: #y axis
        rot_mat = np.array([[math.cos(angle), 0, math.sin(angle)],
                             [0, 1, 0],
                             [-math.sin(angle), 0, math.cos(angle)]])
    else : # z axis
        rot_mat = np.array([[math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0],
                              [0, 0, 1]])
    
    for i in range(rotated_points.shape[0]):
        rotated_points[i,:] = np.dot(rot_mat, rotated_points[i,:])
    return(rotated_points)


def centering_origin(points, centroid):
    """center the point cloud on the origin"""
    centred_points = points.copy()
    centred_points = centred_points + centroid
    return(centred_points)

def translating_y(points, dist):
    """make a translation along y axis"""
    tr = [0, dist, 0]
    translated_points = points.copy()
    translated_points = translated_points + tr
    return(translated_points)

def noise(points, offset):
    """add noise to the point cloud"""
    noised_points = points.copy()
    axis_list = np.random.randint(0, 3, noised_points.shape[0])
    for i in range(len(noised_points)):
        noised_points[i,axis_list[i]] += offset
    return(noised_points)

# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    nb_experiments = 10
    angles_list = 2*math.pi*np.random.random(nb_experiments)
    axis_list = np.random.randint(0, 3, nb_experiments)
    scale_range = [0.5, 2.0]
    scales_list = (scale_range[1]-scale_range[0])*np.random.random(nb_experiments)+scale_range[0]
    offset_range = [10**(-4), 10**(-3)]
    
    # Load point cloud
    file_path = 'C://Users//juliette//Desktop//enpc//3A//S2//Nuage_de_points_et_modélisation_3D//projet//github//data//bunny_normals.ply' # Path of the file
    data = read_ply(file_path)
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # rotation
    for i in range(nb_experiments):
        points_rotated, centroid = centering_centroid(points)
        points_rotated = rotation(points_rotated, angles_list[i], axis_list[i])
        points_rotated = centering_origin(points_rotated, centroid)
        # Save point cloud
        write_ply('../bunny_rotation_angle_'+str(angles_list[i])+'_axis_'+str(axis_list[i])+'.ply', [points_rotated], ['x', 'y', 'z'])

    # scaling
    for scale in scales_list:
        points_scaled, centroid = centering_centroid(points)
        points_scaled = scaling(points_scaled, scale)
        points_scaled = centering_origin(points_scaled, centroid)
        # Save point cloud
        write_ply('../bunny_scaling_'+str(scale)+'.ply', [points_scaled], ['x', 'y', 'z'])

    #noise
    for i in range(nb_experiments):
        offset = offset_range[0] + i*(offset_range[1]-offset_range[0])/nb_experiments
        points_noised, centroid = centering_centroid(points)
        points_noised = noise(points_noised, offset)
        points_noised = centering_origin(points_noised, centroid)
        # Save point cloud
        write_ply('../bunny_noise_'+str(offset)+'.ply', [points_noised], ['x', 'y', 'z'])

    print('Done')
