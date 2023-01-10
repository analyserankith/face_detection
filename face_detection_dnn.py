import numpy as np
import cv2
#%%
img = cv2.imread('web image.jpg')
#%%
model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt','./models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
#%%
blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,223),swapRB=True)
#%%
model.setInput(blob)
detections = model.forward()
image = img.copy()
h , w = image.shape[:2]
for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence >= 0.4:
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        box = box.astype('int')
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),3)
cv2.imshow('face detections',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%