import time
import hashlib
import numpy as np
from foolbox.models import Model


class FrozenNoiseTensorFlowModel(Model):
    def __init__(
            self,
            images,
            logits,
            noises,
            noise_std,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1)):

        super(FrozenNoiseTensorFlowModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self.noise_std = noise_std

        # delay import until class is instantiated
        import tensorflow as tf

        session = tf.get_default_session()
        if session is None:
            session = tf.Session(graph=images.graph)
            self._created_session = True
        else:
            self._created_session = False

        with session.graph.as_default():
            self._session = session
            self._images = images
            self._batch_logits = logits
            self._noises = noises
            self._logits = tf.squeeze(logits, axis=0)
            self._label = tf.placeholder(tf.int64, (), name='label')

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._label[tf.newaxis],
                logits=self._logits[tf.newaxis])
            self._loss = tf.squeeze(loss, axis=0)
            gradients = tf.gradients(loss, images)
            assert len(gradients) == 1
            if gradients[0] is None:
                gradients[0] = tf.zeros_like(images)
            self._gradient = tf.squeeze(gradients[0], axis=0)

            self._bw_gradient_pre = tf.placeholder(tf.float32, self._logits.shape)  # noqa: E501
            bw_loss = tf.reduce_sum(self._logits * self._bw_gradient_pre)
            bw_gradients = tf.gradients(bw_loss, images)
            assert len(bw_gradients) == 1
            if bw_gradients[0] is None:
                bw_gradients[0] = tf.zeros_like(images)
            self._bw_gradient = tf.squeeze(bw_gradients[0], axis=0)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._created_session:
            self._session.close()
        return None

    @property
    def session(self):
        return self._session

    def num_classes(self):
        _, n = self._batch_logits.get_shape().as_list()
        return n

    def get_noise_seed(self, x):
        assert x.shape == (64, 64, 3)
        assert x.dtype == np.float32

        b = x.tobytes()
        h = hashlib.sha256(b)
        hd = h.digest()
        s = int.from_bytes(hd, byteorder='big') % 2**32
        return s

    def get_noise_feed_dict(self, images):
        N = len(images)
        noise_shapes = []
        for noise_ph in self._noises:
            shape = tuple(noise_ph.shape.as_list())
            assert shape[0] is None
            noise_shapes.append((N,) + shape[1:])
        noise_values = [np.zeros(shape, dtype=np.float32)
                        for shape in noise_shapes]
        for i in range(N):
            seed = self.get_noise_seed(images[i])
            np.random.seed(seed)
            for shape, array in zip(noise_shapes, noise_values):
                assert array[i].shape == shape[1:]
                array[i] = np.random.normal(scale=self.noise_std, size=shape[1:])

        noise_dict = {}
        for placeholder, value in zip(self._noises, noise_values):
            noise_dict[placeholder] = value

        return noise_dict

    def batch_predictions(self, images):
        start = time.time()
        feed_dict = self.get_noise_feed_dict(images)
        duration = time.time() - start
        N = len(images)
        print(f'producing noise for {N} images took'
              f' {duration:.1f}ms, {duration / N:.1f}ms per image')

        images = self._process_input(images)
        feed_dict[self._images] = images

        predictions = self._session.run(
            self._batch_logits,
            feed_dict=feed_dict)
        return predictions
