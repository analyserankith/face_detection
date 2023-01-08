import numpy as np
import cv2
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
vid = cv2.VideoCapture(0)
while True:
    check , frames = vid.read()
    if check == False:
        break
    cv2.imshow('face detection',face_detection(frames))
    if cv2.waitKey(1) == ord('1'):
        break
vid.release()
cv2.destroyAllWindows()
#%%