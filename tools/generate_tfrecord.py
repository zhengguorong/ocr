# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from os import walk
from os.path import join
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

image_size = 32
num_labels = len(open('dataset/y_tag.txt').readlines())

def read_images(path):
  file_object = open(path)
  # 逐行读取txt文件，images/10/6_NotoSansHans-Black.otf.jpg 10
  list_of_all_the_lines = file_object.readlines()
  # 训练集大小
  train_num = len(list_of_all_the_lines)
  images = np.zeros((train_num, image_size , image_size), dtype=np.uint8)
  labels = np.zeros((train_num, ), dtype=np.uint8)
  for i, line in enumerate(list_of_all_the_lines):
    line=line.strip('\n')
    filds = line.split(' ')
    file_path = 'dataset/' + filds[0]
    img = imread(file_path)
    images[i] = img
    labels[i] = filds[1]
  return images, labels

def covert(images, labels, save_path):
  num = images.shape[0]
  writer = tf.python_io.TFRecordWriter(save_path)
  for i in range(num):
    img_raw = images[i].tostring()
    label = labels[i]
    example = tf.train.Example(features=tf.train.Features(feature={
      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
      'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))
    writer.write(example.SerializeToString())
  writer.close()
  print 'successfuly write tf file'

def read_and_decode(filename):
  filename_queue = tf.train.string_input_producer([filename])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features={
    'label': tf.FixedLenFeature([], tf.int64),
    'img_raw': tf.FixedLenFeature([], tf.string)
  })
  img = tf.decode_raw(features['img_raw'], tf.uint8)
  img = tf.reshape(img, [image_size * image_size])
  img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
  label = tf.cast(features['label'], tf.int32)
  return img, label

def read_tf(file):
  img, label = read_and_decode(file)
  img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=20, capacity=2000, min_after_dequeue=1000)
  init = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    print img_batch
    for i in range(1):
      val, l = sess.run([img_batch, label_batch])
      print(l[0])
      test_label = tf.one_hot(l[0], num_labels, on_value=1)
      print(test_label.eval())
      plt.imshow(val[0].reshape(32, 32))
      plt.show() 

def main(_):
  test_images, test_labels = read_images('dataset/test.txt')
  covert(test_images, test_labels, 'dataset/test.record')
  train_images, train_labels = read_images('dataset/train.txt')
  covert(train_images, train_labels, 'dataset/train.record')

if __name__ == '__main__':
  tf.app.run()
