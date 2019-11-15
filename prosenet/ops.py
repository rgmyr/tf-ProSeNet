"""
Pure TensorFlow functions
"""
import tensorflow as tf

def make2d(t):
    """Make a Tensor `t` 2D, raise ValueError if impossible."""
    ndim = tf.ndim(t)
    if ndim == 2:
        return t
    elif ndim == 1:
        return tf.expand_dims(t, 0)
    else:
        t = tf.squeeze(t)
        if tf.ndim(t) != 2:
            raise ValueError()
        else:
            return t


def distance_matrix(a, b):
    """Return the distance matrix between rows of `a` and `b`

    They must both be squeezable or expand_dims-able to 2D.
    """
    pass
