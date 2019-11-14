"""
A `Prototype` Layer and related operations.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer

class Prototype(KerasLayer):
    """
    Should we combine this with a (positive restricted) Linear classifier, or handle that in a full model?
    """
    def __init__(self, k, **kwargs):
        """
        Parameters
        ----------
        k : int
            Number of prototype vectors to create.
        """
        super(Prototype, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):

        # what initializer should we use?
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.k, input_shape[-1]),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x):

        # L2 distances from prototypes
        d2 = tf.norm(x - self.prototypes, ord=2, axis=-1)

        # return exponentially squashed
        return tf.exp(-d2)
