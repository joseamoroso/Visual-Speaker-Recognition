
from __future__ import division
import cv2



#Detectar rostro usando redes neuronales
def detectFaceOpenCVDnn(net, frame):
    conf_threshold = 0.7
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes# -*- coding: utf-8 -*-

def detectFaceViolaJ(frame):
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('D:/Tesis/visualspeakerecognition/auxiliars/haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    for (column, row, width, height) in detected_faces:
        new_Frame = frame[row:row+height,column:column+width]
    return new_Frame

# img = cv2.imread("C:/Users/josel/Desktop/imagen1.png")
# a = detectFaceViolaJ(img)
# cv2.imwrite("cara1.jpg",a)



    

