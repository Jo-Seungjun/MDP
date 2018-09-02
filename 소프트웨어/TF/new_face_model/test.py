import tensorflow as tf

x = tf.placeholder(tf.float32,shape=[None,5])
y = tf.placeholder(tf.float32,shape=[None,5])
softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=x,labels=y)

with tf.Session() as sess:
    feedx=[[0.1,0.2,0.3,0.4,0.5],[0.,0.,0.,0.,1.]]
    feedy=[[1.,0.,0.,0.,0.],[0.,0.,0.,0.,1.]]
    softmax = sess.run(softmax, feed_dict={x:feedx, y:feedy})
    print("softmax", softmax)

