"""
A `Prototypes` Layer and related operations.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer


class Prototypes(KerasLayer):
    """
    Should we combine this with a (positive restricted) Linear classifier,
    or handle that in a full model?

    Do we add losses in `call`?
    """
    def __init__(self, k, dmin=1.0, **kwargs):
        """
        Parameters
        ----------
        k : int
            Number of prototype vectors to create.
        dmin : float, optional
            Threshold to determine whether to prototypes are close, default=1.0.
            For "diversity" regularization. See paper section 3.2 for details.
        """
        super(Prototype, self).__init__(**kwargs)
        self.k = k
        self.dmin = dmin


    def build(self, input_shape):

        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(1, self.k, input_shape[-1]),
            initializer='random_normal',        # is this the right init?
            trainable=True
        )


    def call(self, x):

        # L2 distances from prototypes
        x = tf.expand_dims(x, -2)
        d2 = tf.norm(x - self.prototypes, ord=2, axis=-1)

        return tf.exp(-d2)


    def _diversity_loss(self):
        # Compute the "diversity" loss,
        # which penalizes prototypes that are close to each other
        pass

    def _clustering_loss(self, d2):
        # Compute the "clustering" loss,
        # which minimizes distance between encodings and nearest prototypes
        pass

    def _evidence_loss(self, d2):
        # Compute the "evidence" loss,
        # which pushes each prototype to be close to an encoding
        pass