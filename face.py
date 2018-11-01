import numpy as np
import cv2
import pickle

face_cascade=cv2.CascadeClassifier('python/facecascade.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

labels={}
with open("face.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
print(labels)
cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id1, conf = recognizer.predict(roi_gray)
        if conf>=30 and conf <= 85:
            print(conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id1]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
