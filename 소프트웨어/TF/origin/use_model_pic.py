import cv2
import tensorflow as tf
import squeezenet
import os
import numpy as np

cap = cv2.imread('./face_photos\Ju_Yechan\Ju_Yechan_006.jpg')

image = tf.placeholder(tf.float32,shape=[None,224,224,3])

# x = tf.reshape(x,shape=[-1,224,224,3])

decode_label = os.listdir('./face_photos')

class netInit(object):
    num_classes=5
    weight_decay=0.1
    batch_norm_decay=0.999
    
net = squeezenet.Squeezenet(netInit)
print('new net',net)
net = net.build(image, is_training=False)
print("build net",net)

chack_point = './chack_point/save'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess,chack_point)

    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    frame = cap

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255
    RGB = np.expand_dims(RGB, axis=0)

    label = sess.run(net, feed_dict={image:RGB})
    print(label)
    label = decode_label[np.argmax(label)]
    print(label)

    cv2.imshow('frame',frame)

    cv2.waitKey(0)
