#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os

from frozen_noise_model_wrapper import FrozenNoiseTensorFlowModel
from resnet18.resnet_model_with_frozen_noise import Model


def create_model():
    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, (None, 64, 64, 3))

        # preprocessing
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        features = images * 255 - tf.constant(_CHANNEL_MEANS)

        model = Model(
            resnet_size=18,
            bottleneck=False,
            num_classes=200,
            num_filters=64,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=0,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=[2, 2, 2, 2],
            block_strides=[1, 2, 2, 2],
            final_size=512,
            version=2,
            data_format=None)

        # list of noise placeholders
        noises = []

        def add_frozen_noise(inputs):
            assert inputs.dtype == tf.float32
            noise = tf.placeholder(tf.float32, inputs.shape)
            noises.append(noise)
            return inputs + noise

        logits = model(features, False, add_frozen_noise)

        print('noises', noises)

        with tf.variable_scope('utilities'):
            saver = tf.train.Saver()

    return graph, saver, images, logits, noises


def create_fmodel(noise_std=0.6):
    graph, saver, images, logits, noises = create_model()
    sess = tf.Session(graph=graph)
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'resnet18', 'checkpoints', 'model')
    saver.restore(sess, tf.train.latest_checkpoint(path))

    with sess.as_default():
        fmodel = FrozenNoiseTensorFlowModel(
            images, logits, noises, noise_std, bounds=(0, 255), preprocessing=(0, 255))
    return fmodel


if __name__ == '__main__':
    # executable for debuggin and testing
    fmodel = create_fmodel()
    print(fmodel)
    np.random.seed(22)
    demo_images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    print(fmodel.batch_predictions(demo_images[:2])[0, :10])
