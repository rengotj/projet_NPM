# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:40:38 2019
@author: juliette

Compute repeatability assuming that all files in the folder "data/data_to_compute_repeatability"
should have the same IP and that the first one is associated to the reference object
"""

#import sys
#sys.path.append("C://Users//juliette//Desktop//enpc//3A//S2//Nuage_de_points_et_modélisation_3D//projet//github//utils")
#from ply import read_ply
from utils.ply import write_ply, read_ply

import numpy as np
import os

data_path = 'data//data_to_compute_repeatability' # Path of the file
file_list = os.listdir(data_path)

ref_data = read_ply(data_path+'//'+file_list[0])
ref_labels = ref_data['labels_cluster']
num_IP = np.sum(ref_labels)
                      
for i in range(1, len(file_list)):
    data = read_ply(data_path+'//'+file_list[i])
    labels = data['labels_cluster']
    arg_diff = np.where(ref_labels!=labels)[0]
    ref_labels[arg_diff] = 0

num_IP_intersec = np.sum(ref_labels)

R = num_IP_intersec/num_IP
print('Repeatability : ', R)