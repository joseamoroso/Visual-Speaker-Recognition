    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:08:14 2020

@author: jose
"""

from __future__ import division
import cv2
import os 
import glob
import numpy as np
import dlib
import json
from auxiliars.lipsExtraction import  lips_segm_HOG
from auxiliars.faceDetection import detectFaceOpenCVDnn

if __name__ == "__main__" :

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("modelsFaceRecognition\shape_predictor_68_face_landmarks.dat")
    modelFile = "modelsFaceRecognition\opencv_face_detector_uint8.pb"
    configFile = "modelsFaceRecognition\opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    conf_threshold = 0.7
    video = "S001_R01_p0.mp4"
    cam = cv2.VideoCapture(video)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (12,12),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cam.read()
    
    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,old_frame)
    for (x, y, w, h) in bboxes:
        f_image = old_frame[y:h,x:w]
    old_frame =  f_image            
    roi,shape = lips_segm_HOG(f_image,predictor)
    old_gray = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)
    
    p0 = np.array(shape)
    p0 = p0.astype(np.float32)
    p0 = p0.reshape(-1,1,2)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    while(True): 
        # reading from frame 
        ret,frame = cam.read() 
        if ret:
            for (x, y, w, h) in bboxes:
               f_image = frame[y:h,x:w]               
            frame_gray = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                f_image = cv2.circle(f_image,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(f_image,mask)
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
    
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        else:
            break

        
  
    # Release all space and windows once done 
        
    cam.release()
    cv2.destroyAllWindows()



