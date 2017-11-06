import tensorflow as tf
from tf_nearest_neighbour import nn_distance


if __name__ == '__main__':
    import numpy as np
    import random
    import time
    random.seed(100)
    np.random.seed(100)
    with tf.Session('') as sess:
        xyz1 = np.random.randn(32, 16384, 3).astype('float32')
        xyz2 = np.random.randn(32, 1024, 3).astype('float32')
        with tf.device('/gpu:0'):
            inp1 = tf.Variable(xyz1)
            inp2 = tf.constant(xyz2)
            dst1, idx1, dst2, idx2 = nn_distance(inp1, inp2)
            print(dst1.shape)
            print(dst2.shape)
            exit()
            loss = tf.reduce_sum(dst1) + tf.reduce_sum(dst2)
            train = tf.train.GradientDescentOptimizer(
                learning_rate=0.05).minimize(loss)
        sess.run(tf.global_variables_initializer())
        t0 = time.time()
        t1 = t0
        best = 1e100
        for i in xrange(200):
            trainloss, _ = sess.run([loss, train])
            newt = time.time()
            best = min(best, newt - t1)
            print(i, trainloss, (newt - t0) / (i + 1), best)
            t1 = newt
