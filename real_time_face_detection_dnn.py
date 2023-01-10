import numpy as np
import cv2
#%%
def face_detection(img):
    blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=True)
    model.setInput(blob)
    detections = model.forward()
    image = img.copy()
    h,w = image.shape[:2]
    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            box = box.astype('int')
            pt1 = (box[0],box[1])
            pt2 = (box[2],box[3])
            cv2.rectangle(image,pt1,pt2,(0,255,0),1)

            text = 'score : {:.0f} %'.format(confidence*100)
            cv2.putText(image,text,pt1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    return image
#%%
vid = cv2.VideoCapture(0)
model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt','./models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
while True:
    check , frames = vid.read()
    if check == False:
        break
    cv2.imshow("face detection",face_detection(frames))
    if cv2.waitKey(1) == ord('1'):
        break
vid.release()
cv2.destroyAllWindows()
