from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import dataset

import matplotlib.pyplot as plt
import numpy as np
import os
import time
# from networks import squeezenet
import model

dataset = dataset.Data_set()
dataset.open('./face_photos',8)
dataset.shuffle()
#dataset.read(30)

#[None,224,224,3]

# x = tf.placeholder(tf.float32,shape=[None,224,224,3])
# y = tf.placeholder(tf.float32,shape=[None,5])
x, y = dataset.read()
# print(x)
x = tf.reshape(x,shape=[-1,224,224,3])

# class netInit(object):
#     num_classes=5
#     weight_decay=0.1
#     batch_norm_decay=0.1

# x = tf.placeholder(tf.float32,shape=[2,224,224,3])

net = model.Squeezenet(5)
# print('new net',net)
net = net.build(x, is_training=True)
# print("build net",net)
# net = tf.reshape(net,[-1,3490*5])

# print("reshape",net)

# #FC5
# W_5 = tf.Variable(tf.truncated_normal([3490*5,3490],stddev=0.1))
# b_5 = tf.Variable(tf.truncated_normal([3490],stddev=0.1))
# full_5 = tf.nn.relu(tf.matmul(net,W_5) + b_5)

# #FC6
# W_6 = tf.Variable(tf.truncated_normal([3490,5],stddev=0.1))
# b_6 = tf.Variable(tf.truncated_normal([5],stddev=0.1))
# net = tf.matmul(full_5,W_6) + b_6

# print("custom",net)
global_step = tf.Variable(0,trainable=False)

# print(net,y)

#softmax = tf.nn.softmax(net)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y))
#cost = tf.reduce_mean(tf.square(net-y))
#train = tf.train.GradientDescentOptimizer(1e-3).minimize(cost,global_step=global_step)
#train = tf.train.AdamOptimizer(1e-3).minimize(cost, global_step=global_step)
train = tf.train.MomentumOptimizer(learning_rate=1e-3,momentum=0.3).minimize(cost, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

chack_point_path = './chack_point/save'
saver = tf.train.Saver()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if os.listdir("./chack_point") != []:
        saver.restore(sess, chack_point_path)

    for epoch in range(1000):
        for batch in range(125):
            # print("start batch",epoch,batch)
            # print(labels,images)
            sess.run(train)
            # print('trainning')

            # image, name, label, _, step, cos, acc= sess.run([x, y, softmax, train, global_step, cost, accuracy])
            # print("epoch", epoch, "Step", step, "image", image, "in", name, "out", label, "Cost:", cos, "Accuracy", acc)
            # saver.save(sess, chack_point_path)

            # plt.figure
            # #plt.title(label[0])
            # plt.title(name[0]+label[0])
            # plt.imshow(image[0])
            # plt.show()

        # if epoch % 5 == 0:
        #     step, cos, acc= sess.run([global_step, cost, accuracy])
        #     print("epoch", epoch, "\tStep:", step, "\tCost:", cos, "\tAccuracy", acc)
        #     saver.save(sess, chack_point_path)
        step, cos, acc= sess.run([global_step, cost, accuracy])
        print("epoch", epoch, "\tStep:", step, "\tCost:", cos, "\tAccuracy", acc)
        saver.save(sess, chack_point_path)
    # labels, images = dataset.read(100)
    # n=len(images)
    # size = 10
    # plt.figure()
    # plt.gca().set_axis_off()
    # im = np.vstack([np.hstack([sess.run(images[np.random.choice(n)]) for i in range(size)]) for i in range(size)])
    # plt.imshow(im)
    # plt.show()

#print(net)
