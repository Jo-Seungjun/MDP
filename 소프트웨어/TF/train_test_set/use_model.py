import os
import cv2
from scripts.label_image import *


cap = cv2.VideoCapture(0)

def label_image(image):
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(image,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)
    print(image.shape,t.shape)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    return "{} {:0.3f}".format(labels[top_k[0]],results[top_k[0]])

def detect_face():
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

    if face_cascade.empty():
	    raise IOError('Unable to load the face cascade classifier xml file')

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame,1)

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_rects = face_cascade.detectMultiScale(gray,1.1,1,minSize=(224,224))

            for (x, y, w, h) in face_rects:
                face = frame[y:y+h,x:x+w]
                #cv2.imshow('ss',face)

                eye_rects = eye_cascade.detectMultiScale(gray[y:y+int(h/2),x:x+w],1.5,1)
                h2 = int(h/2)
                #cv2.imshow('half',gray[y:y+h2,x:x+w])
                for (ex, ey, ew, eh) in eye_rects:
                    cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),2)
                    
                label = label_image(face)

                cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

detect_face()