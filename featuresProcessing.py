#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:18:29 2020

@author: jose
"""
import json 
import math
import matplotlib.pyplot as plt

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
                yi_prime = ((-(xi - x_c) * math.sin(alpha)) + ((yi - y_c)* math.cos(alpha))) / s
                normalize_coord[speaker][speaker_frame].append((xi_prime,yi_prime))
    return normalize_coord



#Recibe un videos y calcula derivadas entre las coordenadas de frames consecutivos
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
        
    
    

f=open("AV_lips_coordinates_v0.txt", "r")    
contents = json.loads(f.read())
f.close()

#diccionario con fshape para cada frame de todos los videos
normalized = normalize_coordinates(contents)

example = normalized['p1_w1_01']
exa1= list(example)

f_shape = derivate(example)





#Prueba
#example = normalized['p3_w1_00']['p3_w1_00_3']
#example2 =  contents['p1_w1_00']['p1_w1_00_3']
#
#x = []
#y = []
#
#for coor in example:
#    x.append(coor[0])
#    y.append(coor[1])
#    
#plt.plot(x, y)
#plt.show()

    
