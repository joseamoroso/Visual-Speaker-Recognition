
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
def lips_segm_HOG(image,predictor,coorlen):
    he, wi, c = image.shape
    rect = dlib.rectangle(0,0,wi,he)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    i,j = face_utils.FACIAL_LANDMARKS_IDXS.get("mouth")
    #Descomentar para devolve roi con puntos y lineas
    # (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    # roi = image[y-5:y+5 + h, x-2:x+2 + w]
    # clone = image.copy()
    # clone = 255 * np.ones(shape=[he, wi, 3])
    # for (x1, y1) in shape[i:i+coorlen]:
    #     cv2.circle(clone, (x1, y1), 1, (0, 255, 0), -1)    
    # pts = np.array(shape[i:i+12])
    # pts = pts.reshape((-1,1,2))
    # cv2.polylines(clone,[pts],True,(0,255,255))
    # pts = np.array(shape[i+12:i+coorlen])
    # pts = pts.reshape((-1,1,2))
    # cv2.polylines(clone,[pts],True,(0,255,255))
    # clone = clone[y-5:y+5 + h, x-2:x+2 + w]
    # clone = resize(clone, width=450, inter=cv2.INTER_CUBIC)
    clone=None

    return clone,shape[i:i+coorlen]


# predictor = dlib.shape_predictor("D:/Tesis/visualspeakerecognition/modelsFaceRecognition/shape_predictor_68_face_landmarks.dat")
# pru = cv2.imread("C:/Users/josel/Desktop/cara1.jpg")

# a,sha=lips_segm_HOG(pru,predictor,20)

# while(True):
#     cv2.imwrite("lipsPoints.jpg",a)
#     if cv2.waitKey(0):
#         break




