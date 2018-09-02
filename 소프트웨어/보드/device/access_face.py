import cv2
import numpy as np

#load
face_cascade = cv2.CascadeClassifier('haar_cascade_files\haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError('Unable to open cascade filter')


def request(frame):
    face_extracted = face_extraction(frame)

    if face_matching():
        return True,id

    return True,id

def face_matching():
    #id = CNN
    return

def face_extraction(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.VideoCapture(0)
    # Run the face detector on the grayscale image
    face_areas = face_cascade.detectMultiScale(gray,1.1,5)

    faces = []
    # For each face that's detected, run the eye detector
    for (x,y,w,h) in face_areas:
        # Extract the color face ROI
        roi_color = frame[y:y+h, ((x+w/2)-h/2):h]
        faces.append(roi_color)

    # Display the output
    cv2.imshow('Detector', frame)

    return faces