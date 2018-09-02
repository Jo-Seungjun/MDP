import cv2
import os

if not os.path.isdir('./train_set/'):
    os.mkdir('./train_set/')

if not os.path.isdir('./test_set/'):
    os.mkdir('./test_set/')

cap = cv2.VideoCapture(0)

def save_face():
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    ScaleFactor = 1.01
    i = 0
    if face_cascade.empty():
	    raise IOError('Unable to load the face cascade classifier xml file')

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame,1)

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_rects = face_cascade.detectMultiScale(gray,ScaleFactor,1,minSize=(224,224))

            if not face_rects == []:
                size = (0,0,0,0)
                for rect in face_rects:
                    if size[2] < rect[2]:
                        size = rect
                (x, y, w, h) = size

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    print(i,' ',h,' ',w)
                    if  h != 0 and w != 0:
                        face = frame[y:y+h, x:x+w]
                        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                        cv2.imwrite('./train_set/face{0}.jpg'.format(i),face)
                        i = i + 1
                        cv2.putText(frame,'saving',(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            cv2.putText(frame,'q:quit s:save scaleFactor : ' + str(ScaleFactor),(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

            key = cv2.waitKey(1) & 0xFF

            if key==0:
                ScaleFactor = ScaleFactor + 0.01
            elif key==1:
                ScaleFactor = ScaleFactor - 0.01

            cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def save_frame():
    i=0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        if ret:
            if cv2.waitKey(2) & 0xFF == ord('s'):
                cv2.imwrite('./train_set/frame{0}.jpg'.format(i),frame)
                i = i + 1
                cv2.putText(frame,'saving',(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            x=int(len(frame[0])/2)
            y=int(len(frame)/2)
            cv2.rectangle(frame,(x-112,y-112),(x+112,y+112),(255,0,0),2)
            cv2.putText(frame,'q:quit s:save',(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

            cv2.imshow('frame',frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    pass

if __name__ == "__main__":

    save_face()
    #save_frame()