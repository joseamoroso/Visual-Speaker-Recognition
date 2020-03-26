#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:18:29 2020

@author: jose
"""
import json 
import math
import matplotlib.pyplot as plt
import numpy as np

NUM_CORD = 12


#Estructura del argumento coor_list:
#| contents    
    #| p4_w1_02
    #    | p4_w1_02_00
    #        |  [[86, 97],
    #            [111, 109],
    #            [132, 119],
    #            [145, 113],
    #            [158, 117],
    #            [178, 108],
    #            [204, 96],
    #            [178, 83],
    #            [158, 79],
    #            [144, 80],
    #            [130, 81],
    #            [110, 85]]            
    #    | p4_w1_02_01
    #        |  [[86, 97],
    #            [111, 109],
    #            [132, 119],
    #            [145, 113],
    #            [158, 117],
    #            [178, 108],
    #            [204, 96],
    #            [178, 83],
    #            [158, 79],
    #            [144, 80],
    #            [130, 81],
    #            [110, 85]]  



def normalize_coordinates(coord_list):
    normalize_coord = {}
    for speaker in coord_list:
        normalize_coord[speaker]={}
        for speaker_frame in coord_list[speaker]:
            normalize_coord[speaker][speaker_frame] = []
            x_l = coord_list[speaker][speaker_frame][0][0]
            x_r = coord_list[speaker][speaker_frame][6][0]
            y_l = coord_list[speaker][speaker_frame][0][1]
            y_r = coord_list[speaker][speaker_frame][6][1]    
            x_c = (x_l + x_r) / 2
            y_c = (y_l + y_r) / 2
            alpha = math.atan2(y_r-y_l,x_r-x_l)
            s = math.sqrt((((x_l-x_r)**2)+((y_l-y_r)**2))/2)
            
            for coord in coord_list[speaker][speaker_frame]:
                xi = coord[0]
                yi = coord[1]
                xi_prime = (((xi - x_c) * math.cos(alpha)) + ((yi - y_c)* math.sin(alpha))) / s
                yi_prime = -(((-(xi - x_c) * math.sin(alpha)) + ((yi - y_c)* math.cos(alpha))) / s)
                normalize_coord[speaker][speaker_frame].append((xi_prime,yi_prime))
    return normalize_coord



#Recibe un video y calcula derivadas entre las coordenadas de frames consecutivos
def derivate (video_frames):
    f_temp_shape = {}
    num_frames = len(video_frames)
    frames_keys = list(video_frames.keys())
    for i in range(num_frames-1):
        f_temp_shape["f_temp_shape"+str(i)]=[]
        for coor in range(NUM_CORD):
            x_act = video_frames[frames_keys[i]][coor][0]
            x_next = video_frames[frames_keys[i+1]][coor][0]
            y_act = video_frames[frames_keys[i]][coor][1]
            y_next = video_frames[frames_keys[i+1]][coor][1]
            x_derivate = (x_next - x_act) / 2
            y_derivate =  (y_next - y_act) / 2
            f_temp_shape["f_temp_shape"+str(i)].append((x_derivate,y_derivate))
    return f_temp_shape

def loop_over_static(norm_dict,key):
    frame_features = norm_dict[key] #Lista de diccionarios de frames
    f_temp_shape_dict = list(frame_features)    
    X = np.array([])
    for f_temp_shape_values in f_temp_shape_dict:
        features_list = frame_features[f_temp_shape_values]
        new_features_list = np.array([j for i in features_list for j in i])
        
        if len(X) == 0:
            X = new_features_list
        else:
            X = np.append(X,new_features_list,axis=0)
        new_features_list = []

    return X.reshape(len(f_temp_shape_dict),24)
        
    
    





############################################################################
    
#Prueba
f=open("AV_lips_coordinates_v0.txt", "r")    
contents = json.loads(f.read())
f.close()

#diccionario con fshape para cada frame de todos los videos
normalized = normalize_coordinates(contents)
example = normalized['S001_R01_p0']['S001_R01_p0_10']
example_X = loop_over_static(normalized,'S001_R01_p0')



#x = []
#y = []
#
#for coor in example:
#    x.append(coor[0])
#    y.append(coor[1])
#    
#x.append(x[0])
#y.append(y[0])
#
#    
#plt.plot(x, y)
#plt.show()

    
