
from __future__ import division
import cv2
from imutils import face_utils, resize
import numpy as np
import dlib


#Segmentacion de labios (ALgoritmo de distribucion facial)
def lips_segm_geomtric(image):
    frame_height,frame_width= image.shape[:2]               
    roi = image[((frame_height//3)*2)-15:(frame_height//10)*9,(frame_width//4):(frame_width//4)*3]
    return roi

#Segmentacion de labios (ALgoritmo arboles de regresion)
def lips_segm_HOG(image,predictor):
    he, w, c = image.shape
    rect = dlib.rectangle(0,0,w,he)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    i,j = face_utils.FACIAL_LANDMARKS_IDXS.get("mouth")
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = image[y-5:y+5 + h, x-2:x+2 + w]
    clone = image.copy()
    for (x1, y1) in shape[i:i+12]:
        cv2.circle(clone, (x1, y1), 2, (0, 255, 0), -1)    
    pts = np.array(shape[i:i+12])
    pts = pts.reshape((-1,1,2))
    cv2.polylines(clone,[pts],True,(0,255,255))
    clone = clone[y-5:y+5 + h, x-2:x+2 + w]
    clone = resize(clone, width=450, inter=cv2.INTER_CUBIC)

    return clone,shape[i:i+12]