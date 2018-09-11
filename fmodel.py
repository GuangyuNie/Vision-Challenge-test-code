#!/usr/bin/env python3
import tensorflow as tf
from lenet_model import LeNet
import numpy as np
import foolbox


def create_model(): 
    images = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
    preprocessed = images - [123.68, 116.78, 103.94]
    logits, _ = LeNet(preprocessed)
    restorer = tf.train.Saver()


    with tf.Session() as sess:
        restorer.restore(sess, 'lenet.ckpt')
        print('model restored')
        fmodel = foolbox.models.TensorFlowModel(images, logits, (0, 255))

    return fmodel
