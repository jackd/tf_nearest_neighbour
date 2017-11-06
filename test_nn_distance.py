import unittest
import numpy as np
import tensorflow as tf
from tf_nearest_neighbour import nn_distance


def simple_nn(xyz1, xyz2):
    def is_valid_shape(shape):
        return len(shape) == 3 and shape[-1] == 3
    assert(is_valid_shape(xyz1.shape))
    assert(is_valid_shape(xyz2.shape))
    assert(xyz1.shape[0] == xyz2.shape[0])
    diff = np.expand_dims(xyz1, -2) - np.expand_dims(xyz2, -3)
    square_dst = np.sum(diff**2, axis=-1)
    dst1 = np.min(square_dst, axis=-1)
    dst2 = np.min(square_dst, axis=-2)
    idx1 = np.argmin(square_dst, axis=-1)
    idx2 = np.argmin(square_dst, axis=-2)
    return dst1, idx1, dst2, idx2


def tf_nn(xyz1, xyz2, device):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device):
            xyz1 = tf.constant(xyz1)
            xyz2 = tf.constant(xyz2)
            nn = nn_distance(xyz1, xyz2)

    with tf.Session(graph=graph) as sess:
        actual = sess.run(nn)
    return actual


devices = ['/cpu:0', '/gpu:0']


class TestNnDistance(unittest.TestCase):

    def _compare_values(self, actual, expected):

        self.assertEqual(len(actual), len(expected))
        # distances
        for i in [0, 2]:
            np.testing.assert_allclose(actual[i], expected[i])
        # indices
        for i in [1, 3]:
            np.testing.assert_equal(actual[i], expected[i])

    def _compare(self, xyz1, xyz2, expected):
        for device in devices:
            actual = tf_nn(xyz1, xyz2, device)
            self._compare_values(actual, expected)

    def test_small(self):
        xyz1 = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        xyz2 = np.array([[[-100, 0, 0], [2, 0, 0]]], dtype=np.float32)
        expected = \
            np.array([[4, 1, 5]]), \
            np.array([[1, 1, 1]]), \
            np.array([[10000, 1]]), \
            np.array([[0, 1]])
        self._compare(xyz1, xyz2, expected)

    def test_big(self):
        batch_size = 5
        n1 = 10
        n2 = 20
        xyz1 = np.random.randn(batch_size, n1, 3).astype(np.float32)
        xyz2 = np.random.randn(batch_size, n2, 3).astype(np.float32)
        expected = simple_nn(xyz1, xyz2)
        self._compare(xyz1, xyz2, expected)


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    unittest.main()
