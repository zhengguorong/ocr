# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
sys.path.append('tools')
import tempfile
import tensorflow as tf
from generate_tfrecord import read_and_decode
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imread, imresize

FLAGS = None

SUMMARY_DIR = "./log/supervisor.log"
train_file = 'dataset/train.record'
test_file = 'dataset/test.record'
image_size = 32
num_labels = 6493
num_channels = 1
batch_size = 128


def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, image_size, image_size, 1])
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 128, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, num_labels])
        b_fc2 = bias_variable([num_labels])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    x = tf.placeholder(tf.float32, [None, image_size * image_size])
    y_ = tf.placeholder(tf.float32, [None, num_labels])
    y_conv, keep_prob = deepnn(x)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    train_img, train_label = read_and_decode(train_file)
    train_label = tf.one_hot(train_label, num_labels, on_value=1)
    test_img, test_label = read_and_decode(test_file)
    test_label = tf.one_hot(test_label, num_labels, on_value=1)
    train_img_batch_tensor, train_label_batch_tensor = tf.train.shuffle_batch(
        [train_img, train_label], batch_size=batch_size, capacity=20000, min_after_dequeue=5000)
    test_img_batch_tensor, test_label_batch_tensor = tf.train.shuffle_batch(
        [test_img, test_label], batch_size=100, capacity=10000, min_after_dequeue=5000)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        train_img_batch, train_label_batch = sess.run(
            [train_img_batch_tensor, train_label_batch_tensor])
        test_img_batch, test_label_batch = sess.run(
            [test_img_batch_tensor, test_label_batch_tensor])
        for i in range(200000):
            if i % 50000 == 0:
                saver.save(sess, "./crack_capcha.model")
            if i % 100 == 0:
                summary, train_accuracy, loss = sess.run([merged, accuracy, cross_entropy], feed_dict={
                                                         x: test_img_batch, y_: test_label_batch, keep_prob: 1})
                summary_writer.add_summary(summary, i)
                print('Step %d: loss = %g Validation accuracy = %.1f%%' %
                      (i, loss, train_accuracy * 100))
            summary, _ = sess.run([merged, train_step], feed_dict={
                                  x: train_img_batch, y_: train_label_batch, keep_prob: 0.6})


def predict(file):
    x = tf.placeholder(tf.float32, [None, image_size * image_size])
    y_conv, keep_prob = deepnn(x)
    img, label = read_and_decode(file)
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], batch_size=20, capacity=2000, min_after_dequeue=1000)
    predict = tf.argmax(y_conv, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        threads = tf.train.start_queue_runners(sess=sess)
        val, l = sess.run([img_batch, label_batch])
        predict = sess.run(predict, feed_dict={x: [val[0]], keep_prob: 1})
        print("预测值：{}, 正确值：{}".format(l[0], predict[0]))
        # plt.imshow(val[0].reshape(32, 32))
        # plt.show()


if __name__ == '__main__':
    tf.app.run()
    # predict(test_file)
