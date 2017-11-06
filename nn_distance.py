import os
import tensorflow as tf
from tensorflow.python.framework import ops
_op_library = tf.load_op_library(
    os.path.join(os.path.dirname(__file__), 'src', 'tf_nndistance.so'))


def nn_distance(xyz1, xyz2):
    """
    Computes the distance of nearest neighbors for a pair of point clouds.

    Args:
        xyz1: (batch_size, n1, 3) the first point cloud
        xyz2: (batch_size, n2, 3) the second point cloud

    Returns:
        (dist1, idx1, dist2, idx2)
        dist1: (batch_size, n1)  squared distance from first to second
        idx1:  (batch_size, n1)  nearest neighbor from first to second
        dist2: (batch_size, n2)  squared distance from second to first
        idx2:  (batch_size, n2)  nearest neighbor from second to first
    """
    return _op_library.nn_distance(xyz1, xyz2)


@ops.RegisterShape('NnDistance')
def _nn_distance_shape(op):
    shape1 = op.inputs[0].get_shape().with_rank(3)
    shape2 = op.inputs[1].get_shape().with_rank(3)
    return [tf.TensorShape([shape1.dims[0], shape1.dims[1]]),
            tf.TensorShape([shape1.dims[0], shape1.dims[1]]),
            tf.TensorShape([shape2.dims[0], shape2.dims[1]]),
            tf.TensorShape([shape2.dims[0], shape2.dims[1]])]


@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op, grad_dist1, grad_idx1, grad_dist2, grad_idx2):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    idx1 = op.outputs[1]
    idx2 = op.outputs[3]
    return _op_library.nn_distance_grad(
        xyz1, xyz2, grad_dist1, idx1, grad_dist2, idx2)
