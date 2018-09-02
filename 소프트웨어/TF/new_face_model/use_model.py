import cv2
import tensorflow as tf
import model
import os
import numpy as np

cap = cv2.VideoCapture('video.mp4')

image = tf.placeholder(tf.float32,shape=[None,224,224,3])

# x = tf.reshape(x,shape=[-1,224,224,3])

decode_label = os.listdir('./face_photos')

net = model.Squeezenet(5)
print('new net',net)
net = net.build(image, is_training=False)
print("build net",net)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

chack_point = './chack_point/save'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess,chack_point)

    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame,1)

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            face_rects = face_cascade.detectMultiScale(gray,1.1,1,minSize=(224,224))
            
            for (x, y, w, h) in face_rects:
                face = frame[y:y+h,x:x+w]
                
                face = cv2.resize(face, (224,224), interpolation=cv2.INTER_CUBIC)
                RGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)/255
                RGB = np.expand_dims(RGB, axis=0)
                # print(RGB.dtype)
                # # RGB = tf.convert_to_tensor(RGB,tf.float32)
                # print(image)
                # print(RGB)
                # print(net)
                #cv2.imshow('face',RGB)
                
                #cv2.imshow('ss',face)

                # eye_rects = eye_cascade.detectMultiScale(gray[y:y+int(h/2),x:x+w],1.5,1)
                # h2 = int(h/2)
                # #cv2.imshow('half',gray[y:y+h2,x:x+w])
                # for (ex, ey, ew, eh) in eye_rects:
                #     cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),2)
                
                label = sess.run(net, feed_dict={image:RGB})
                # print(label)
                label = decode_label[np.argmax(label)]

                cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
