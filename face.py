import numpy as np
import cv2
#%%
img = cv2.imread('web image.jpg')
#%%
def face_detection(img):
    img_copy = img.copy()
    cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
    grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    box, detections = cascade.detectMultiScale2(grey_img,minNeighbors = 8)
    for x,y,w,h in box:
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0))
    return img_copy
#%%
def display_detections(img):
    face_detections = face_detection(img)
    cv2.imshow('face detected image',face_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#%%
display_detections(img)
#%%