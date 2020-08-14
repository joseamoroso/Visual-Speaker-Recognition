#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:08:14 2020

@author: jose
"""

from __future__ import division
import cv2
import glob
import dlib
import json
from auxiliars.lipsExtraction import  lips_segm_HOG
from auxiliars.faceDetection import detectFaceOpenCVDnn,detectFaceViolaJ

#change to 12
NCOORDINATES = 12

if __name__ == "__main__" :
    datasetMode = ["Normal","Silent","Whispered"]
    predictor = dlib.shape_predictor("modelsFaceRecognition\shape_predictor_68_face_landmarks.dat")
    modelFile = "modelsFaceRecognition\opencv_face_detector_uint8.pb"
    configFile = "modelsFaceRecognition\opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    for mode in datasetMode:

        text_filename = "LipsCoordinates_"+ mode +"_12coor_Phrases_ViolaJ.txt"
    
        # extracted_frames_path = "LipsFrames\\"
        videos_path = r"AVSegmentedDataset\Phrases" +"\\"+ mode +"\*.mp4" 
        videos = glob.glob(videos_path)
        currentVideo = 0

        speakerNameList = []
        speakerNameDict = {}
        
        counter=0   
        for video in videos:
            counter+=1
            name_v = video.split('\\')
            name_v = name_v[3].split('.')
            currentVideo=+1
            cam = cv2.VideoCapture(video) 
            if name_v[0] not in speakerNameDict:
                speakerNameList.append(name_v[0])
                speakerNameDict[name_v[0]]={}

            currentframe = 0
            c = 0
            print(str(counter) + ' of ' + str(len(videos)) + '\n')
    
            while(True): 
                # reading from frame 
                ret,frame = cam.read() 
              
                if ret:
##############################################################################
########################### DESCOMENTAR PARA USAR RED NEURONAL ###############
##############################################################################
                    
                    # #Deteccion del rostro usando la red definida previamente
                    # outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
                    # #Recorte del rostro de la imagen original
                    # for (x, y, w, h) in bboxes:
                    #     f_image = frame[y:h,x:w]
                        
##############################################################################
########################## DESCOMENTAR PARA USAR VIOLA JONES #################
##############################################################################

                    try:   
                        f_image = detectFaceViolaJ(frame)   
                    except:
                        # print("error")
                        continue
                        
##############################################################################
########### EXTRACCION DE COORDENADAS SOBRE PARTE DEL ROSTRO #################
##############################################################################
                    roi,shape = lips_segm_HOG(f_image,predictor,NCOORDINATES)
                    speakerNameDict[name_v[0]][name_v[0]+'_' +str(currentframe)]=shape.tolist()                
                  
                    currentframe += 1
    
                    
                else: 
                    break
              
            # Release all space and windows once done 
            cam.release()
        points_json = json.dumps(speakerNameDict)
        f= open(text_filename,"w+")
        f.write(points_json)
        f.close()  
    


