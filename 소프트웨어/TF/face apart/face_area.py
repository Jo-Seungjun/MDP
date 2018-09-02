import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')
 
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.6
font_bold = 1

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray,1.1,1,minSize=(224,224))
        for (x, y, w, h) in face_rects:
            
                cv2.putText(gray,'face',(x,y),font,font_size,(0,0,0),font_bold)
                cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,0),1)

                face = gray[y:y+h,x:x+w]

                h2 = int(h/2)
                h4 = int(h/4)
                w2 = int(w/2)
                w4 = int(w/4)

                #top_face = face[0:h2,0:w]
                top_left_face = face[0:h2,0:w2]
                top_right_face = face[0:h2,w2:-1]
                center_face = face[h4:h4+h2,w4:w4+w2]
                bottom_face = face[h2:-1,0:w]
                left_face = face[0:-1,0:w2]
                right_face = face[0:-1,w2:-1]
                
                left_eye_rects = eye_cascade.detectMultiScale(top_left_face,1.5,1)
                right_eye_rects = eye_cascade.detectMultiScale(top_right_face,1.5,1)
                nose_rects = nose_cascade.detectMultiScale(center_face,1.5,1)
                mouth_rects = mouth_cascade.detectMultiScale(bottom_face,1.5,1)

                for (x, y, w, h) in left_eye_rects:
                    x = x + 0
                    y = y + 0
                    cv2.putText(face,'Left_eye',(x,y),font,font_size,(0,0,0),font_bold)
                    cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,0),1)

                for (x, y, w, h) in right_eye_rects:
                    x = x + w2
                    y = y + 0
                    cv2.putText(face,'Right_eye',(x,y),font,font_size,(0,0,0),font_bold)
                    cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,0),1)
                
                for (x, y, w, h) in nose_rects:
                    x = x + w4
                    y = y + h4
                    cv2.putText(face,'Nose',(x,y),font,font_size,(0,0,0),font_bold)
                    cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,0),1)

                for (x, y, w, h) in mouth_rects:
                    x = x + 0
                    y = y + h2
                    cv2.putText(face,'Mouth',(x,y),font,font_size,(0,0,0),font_bold)
                    cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,0),1)
                

                #cv2.imshow('top_left_face',top_left_face)
                #cv2.imshow('top_right_face',top_right_face)
                #cv2.imshow('face',face)
                #cv2.imshow('top',top_face)
                #cv2.imshow('middle',middle_face)
                #cv2.imshow('bottom',bottom_face)
                #cv2.imshow('left',left_face)
                #cv2.imshow('right',right_face)
        cv2.imshow('gray',gray)

    cv2.waitKey(1)