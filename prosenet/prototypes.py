"""
A `Prototypes` Layer and related operations.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer


class Prototypes(KerasLayer):
    """
    Should we combine this with a (positive restricted) Linear classifier,
    or handle that in a full model?
        - Probably in model, where there can also be interpretation methods

    Do we add losses in `call`?
        - Yes, I think so. But only if training? Or always computed?
    """
    def __init__(self, k, dmin=1.0, Ld=0.01, Lc=0.01, Le=0.1, **kwargs):
        """
        Parameters
        ----------
        k : int
            Number of prototype vectors to create.
        dmin : float, optional
            Threshold to determine whether to prototypes are close, default=1.0.
            For "diversity" regularization. See paper section 3.2 for details.
        Ld : float, optional
            Weight for "diversity" regularization loss, default=0.01.
        Lc : float, optional
            Weight for "clustering" regularization loss, default=0.01.
        Le : float, optional
            Weight for "evidence" regularization loss, default=0.1.
        **kwargs
            Additional arguments for base `Layer` constructor (name, etc.)
        """
        super(Prototype, self).__init__(**kwargs)
        self.k = k
        self.dmin = dmin
        self.Ld, self.Lc, self.Le = Ld, Lc, Le


    def build(self, input_shape):
        # Create prototypes as variable

        normal_init = tf.random_normal_initalizer()
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(1, self.k, input_shape[-1]),
            initializer=normal_init,    # is this the right init?
            trainable=True
        )


    def call(self, x, training=None):
        """Forward pass."""

        # Losses only computed `if training`
        if training:
            if self.Ld > 0.:
                self.add_loss(self.Ld * self._diversity_term())


        # L2 distances from prototypes
        x = tf.expand_dims(x, -2)
        d2 = tf.norm(x - self.prototypes, ord=2, axis=-1)

        return tf.exp(-d2)


    def _diversity_term(self):
        """Compute the "diversity" loss,
        which penalizes prototypes that are close to each other

        NOTE: Computes full distance matrix, which is redudant, but prototypes is
              usually a small-ish tensor, so I'm not going to worry about it.
        """
        p = tf.squeeze(self.prototypes)

        r = tf.expand_dims(tf.reduce_sum(p*p, 1), -1)

        D = r - 2 * tf.matmul(p, p, transpose_b=True) + tf.transpose(r)

        Rd = tf.nn.relu(tf.sqrt(D) - self.dmin)

        return tf.reduce_sum(Rd) / 2.


    def _clustering_evidence_terms(self, x):
        """ Compute the "clustering" loss,
        which minimizes distance between encodings and nearest prototypes
            And the "evidence" loss,
        which pushes each prototype to be close to an encoding
        """
        p = tf.squeeze(self.prototypes)

        r = tf.expand_dims


    def _evidence_term(self, d2):
        # Compute the "evidence" loss,
        # which pushes each prototype to be close to an encoding
        pass


    def get_config(self):
        # implement to make serializable
        pass
