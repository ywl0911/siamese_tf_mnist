# coding:utf8
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

# import system things
from tensorflow.examples.tutorials.mnist import input_data  # for data
import tensorflow as tf
import numpy as np
import os
import sys
import visualize

sys.path.append(r'/home/ywl/Documents/python/siamese_tf_mnist')
os.chdir(r'/home/ywl/Documents/python/siamese_tf_mnist')
# import helpers
from siamese_tf_mnist import inference

# prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
load = False
model_ckpt = 'model_save/model_cnn.meta'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['y', 'n']:
        input_var = input("We found model files. Do you want to load it and continue training [y/n]?")
    if input_var == 'y':
        load = True

# start training
if load:
    saver.restore(sess, 'model_save/model_cnn')
mnist.train.epochs_completed
for step in range(10):
    batch_x1, batch_y1 = mnist.train.next_batch(128)
    batch_x2, batch_y2 = mnist.train.next_batch(128)
    batch_y = (batch_y1 == batch_y2).astype('float')

    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
        siamese.x1: batch_x1,
        siamese.x2: batch_x2,
        siamese.y_: batch_y})

    # if np.isnan(loss_v):
    #     print('Model diverged with loss = NaN')
    #     quit()

    if step % 10 == 0:
        print('step %d: loss %.3f' % (step, loss_v))
        #
        # o1 = siamese.o1.eval({siamese.x1: mnist.test.images[:100]})
        # o2 = siamese.o2.eval({siamese.x2: mnist.test.images[:100]})
        # print(o1[1][:10]==o2[1][:10])

    if step % 1000 == 0 and step > 0:
        saver.save(sess, 'model_save/model_cnn')
        embed = siamese.o1.eval({siamese.x1: mnist.test.data})
        embed.tofile('model_save/embed.txt')

# visualize result
embed1 = siamese.o1.eval({siamese.x1: mnist.test.images[:100]})
embed2 = siamese.o2.eval({siamese.x2: mnist.test.images[:100]})

print(embed1==embed2)
x_test = mnist.test.images.reshape([-1, 28, 28])
y_test = mnist.test.labels
visualize.visualize(embed, x_test, y_test)

# test result
x_train = mnist.train.images
y_train = mnist.train.labels
# 每个类取100张图片作为对比
amount = 100

x_train_last = x_train[y_train == 0][:amount]
y_train_last = [0] * amount
for i in range(1, 10):
    x_train_last = np.vstack((x_train_last, x_train[y_train == i][:amount]))
    y_train_last = y_train_last + [i] * amount

embed_test = siamese.o1.eval({siamese.x1: mnist.test.images})
embed_train = siamese.o1.eval({siamese.x1: x_train_last})
embed_train = np.array(embed_train).reshape(-1, amount, 2)

y_pred = []
for i in embed_test:
    temp_result = []
    for j in range(10):
        dis = np.mean(np.sqrt(np.sum(np.square(i - embed_train[j]), axis=-1)))
        temp_result.append(dis)
    y_pred.append(temp_result.index(min(temp_result)))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
