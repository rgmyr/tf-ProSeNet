"""
Implement a PrototypeProjection callback
"""
from tensorflow.keras.callbacks import Callback


class PrototypeProjection(Callback):
    """
    Implements the "Prototype Projection" step during training.

    Parameters
    ----------
    dataset : prosenet.Dataset
        This callback needs access to the training data.
    freq : int, optional
        How often to call, in epochs, default=4.
    """
    def __init__(self, dataset, freq=4):
        self.dataset = dataset
        self.freq = freq
        super(PrototypeProjection, self).__init__(**kwargs)


    def on_epoch_end(self, epoch, logs=None):
        """
        """
        if epoch % self.freq == 0:
            # get encodings of all train sequences
            
