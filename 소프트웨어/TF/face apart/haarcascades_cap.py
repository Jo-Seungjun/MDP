import cv2
import os
import numpy as np

if not os.path.isdir('./haarcascades/'):
    print('there is no "haar" dir')
    quit()

cascades = []

for haarFileName in os.listdir('./haarcascades'):
    cascade = {'name':haarFileName, 'cascade':cv2.CascadeClassifier('./haarcascades/'+ haarFileName)}
    cascades.append(cascade)

for cascade in cascades:
    print(cascade['cascade'], '\t', cascade['name'])

cap = cv2.VideoCapture(0)

ScaleFactor = 1.01
BlurFactor = 10
HaarFactor = 0
while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame,1)

    if ret:
        #cv2.imshow('image', image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #kernel = np.ones((5,5),np.float32)/25
        #blur = cv2.filter2D(image,-1,kernel)
        blur = cv2.blur(gray,(BlurFactor,BlurFactor))

        temp = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  

        #print(cascades[HaarFactor]['name'])

        cascade = cascades[HaarFactor]['cascade']

        rects = cascade.detectMultiScale(blur,1.2)

        for (x, y, w, h) in rects:
            #cv2.rectangle(temp,(x,y),(x+w,y+h),(0,0,0),1)
            cv2.rectangle(blur,(x,y),(x+w,y+h),(0,0,0),1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5
        font_bold = 1
        #cv2.putText(temp,'ScaleFactor : {0}'.format(str(ScaleFactor)),(0,25),font,1,(0,0,0),2)
        #cv2.putText(temp,'BlurFactor : {0}'.format(str(BlurFactor)),(0,50),font,1,(0,0,0),2)
        cv2.putText(blur,'ScaleFactor : {0}'.format(str(ScaleFactor))                   ,(0,25),font,font_size,(0,0,0),font_bold)
        cv2.putText(blur,'BlurFactor : {0}'.format(str(BlurFactor))                     ,(0,50),font,font_size,(0,0,0),font_bold)
        cv2.putText(blur,'HaarFactor : {0}'.format(str(cascades[HaarFactor]['name']))   ,(0,75),font,font_size,(0,0,0),font_bold)

        #cv2.imshow('Original',gray)
        cv2.imshow('Averaging',blur)
        #cv2.imshow('temp', temp)


        key = cv2.waitKey(3) & 0xFF

        if key==ord('s'):
            ScaleFactor = ScaleFactor + 0.01
        elif key==ord('S'):
            if ScaleFactor > 1.01: ScaleFactor = ScaleFactor - 0.01
        elif key==ord('h'):
            HaarFactor = HaarFactor + 1
        elif key==ord('H'):
            if HaarFactor > 0: HaarFactor = HaarFactor - 1
        elif key==ord('b'):
            BlurFactor = BlurFactor + 1
        elif key==ord('B'):
            if BlurFactor > 1: BlurFactor = BlurFactor - 1
        elif key==ord('q'):
            break

cv2.waitKey(0)