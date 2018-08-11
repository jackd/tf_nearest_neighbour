#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from scipy.spatial import cKDTree
from tf_nearest_neighbour import nn_distance
from time import time


def benchmark(f, n_warmup, n_repeats):
    for i in range(n_warmup):
        f()
    times = np.empty(n_repeats,)
    for i in range(n_repeats):
        t = time()
        f()
        times[i] = time() - t
    return times


class NNImpl(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def __enter__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.i0 = self.build_graph(self.p0, self.p1)
        self.sess = tf.Session(graph=graph)
        return self

    def __exit__(self, *args, **kwargs):
        self.sess.close()

    def build_graph(self, p0, p1):
        raise NotImplementedError()

    def __call__(self):
        return self.sess.run(self.i0)


class CustomOpNN(NNImpl):
    def build_graph(self, p0, p1):
        p0 = tf.constant(self.p0)
        p1 = tf.constant(self.p1)
        d0, i0, d1, i1 = nn_distance(p0, p1)
        return i0


class NaiveNN(NNImpl):
    def build_graph(self, p0, p1):
        p0 = tf.constant(self.p0)
        p1 = tf.constant(self.p1)
        p0 = tf.expand_dims(p0, axis=-2)
        p1 = tf.expand_dims(p1, axis=-3)
        offset = p0 - p1
        dist2 = tf.reduce_sum(offset**2, axis=-1)
        return tf.argmin(dist2, axis=-1)


class CKDTreeNN(NNImpl):
    def __init__(self, p0, p1):
        self.build_trees(p1)
        super(CKDTreeNN, self).__init__(p0, p1)

    def build_trees(self, p1):
        trees = []
        for p1i in p1:
            trees.append(cKDTree(p1i))
        self._trees = trees

    def build_graph(self, p0, p1):
        p0 = tf.constant(self.p0)
        i = tf.range(0, tf.shape(p0)[0], dtype=tf.int32)

        def query_fn(i, p0i):
            dist, i = self._trees[i].query(p0i)
            return i.astype(np.int32)

        def map_fn(inputs):
            # i, p0i = inputs
            return tf.py_func(query_fn, inputs, tf.int32, stateful=True)

        i0 = tf.map_fn(
            map_fn, (i, p0), dtype=tf.int32, back_prop=False,
            infer_shape=False, parallel_iterations=16)
        # i0.set_shape(tf.shape(p0)[:2])
        return i0


class CKDTreeNoMapNN(CKDTreeNN):

    def build_graph(self, p0, p1):
        p0 = tf.constant(self.p0)
        i = tf.range(0, tf.shape(p0)[0], dtype=tf.int32)

        def query_fn(indices, p0):
            nn = np.empty(p0.shape[:2], np.int32)
            for i, p0i in zip(indices, p0):
                dist, ind = self._trees[i].query(p0i)
                nn[i] = ind
            return nn

        i0 = tf.py_func(query_fn, (i, p0), tf.int32, stateful=True)
        return i0


batch_size = 32
n0 = 1024
n1 = 1024*32
p0 = np.random.randn(batch_size, n0, 3).astype(np.float32)
p1 = np.random.randn(batch_size, n1, 3).astype(np.float32)


def benchmark_op(name, op, n_warmup=2, n_repeats=10):
    print('Benchmarking %s' % name)
    with op(p0, p1) as f:
        t = benchmark(f, n_warmup, n_repeats)
    print('Average of %d runs: %.5f' % (n_repeats, np.mean(t)))


def compare_sol(op0, op1):
    with op0(p0, p1) as f0:
        with op1(p0, p1) as f1:
            s0 = f0()
            s1 = f1()
    print(np.all(s0 == s1))


# make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

benchmark_op('cKDTree', CKDTreeNN)
benchmark_op('cKDTreeNpMap', CKDTreeNoMapNN)
benchmark_op('custom', CustomOpNN)
# benchmark_op('naive', NaiveNN)  # this will take a while, may crash GPU

print('CKD impl consistent with custom op:')
compare_sol(CKDTreeNN, CustomOpNN)
print('CKDTree map impl consistent with no-map impl:')
compare_sol(CKDTreeNN, CKDTreeNoMapNN)


# benchmark building trees
impl = CKDTreeNN(p0, p1)


def f():
    impl.build_trees(p1)


print('Average time to build trees:')
t = benchmark(f, 2, 10)
print(np.mean(t))
