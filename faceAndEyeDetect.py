import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,image=cap.read()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in face:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_colour=image[y:y+h,x:x+w]
        eye=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_colour,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)
    cv2.imshow("image",image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
