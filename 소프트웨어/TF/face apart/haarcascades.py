import cv2
import os
import numpy as np

if not os.path.isdir('./haarcascades/'):
    print('there is no "haar" dir')
    quit()
image = cv2.imread('./face.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#kernel = np.ones((5,5),np.float32)/25
#blur = cv2.filter2D(image,-1,kernel)
blur = cv2.blur(gray,(15,15))

cv2.imshow('Original',gray)
cv2.imshow('Averaging',blur)

temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    

for haarFileName in os.listdir('./haarcascades'):
    
    cascade = cv2.CascadeClassifier('./haarcascades/'+ haarFileName)
    
    rects = cascade.detectMultiScale(blur,1.2)
    
    for (x, y, w, h) in rects:
        cv2.rectangle(temp,(x,y),(x+w,y+h),(0,0,0),2)
   
cv2.imshow('name', temp)

while True:
    pass