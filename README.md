# TensorFlow-ProSeNet

This is a `tf.keras` implementation of [Interpretable and Steerable Sequence Learning via Prototypes](https://arxiv.org/abs/1907.09728). Whether I'll implement the "steering" part remains to be seen -- probably not worth it for my purposes.

## Notes

The complete loss minimized is a combination of:

- Cross Entropy
- `lambda_d` * Diversity Regularization on Prototypes (with `d_min`)
- `lambda_e` * Evidence Regularization
- `lambda_c` * Clustering Regularization
- `lambda_l1` * L1 Regularization of Linear Classifier

Additional required operations:

- **Prototype projection**: Every few (~4) epochs, prototype vectors are re-assigned to the closest embedding sequence in the training set.
- **Prototype interpretation**: Need to method to grab classification weights assigned to prototype and interpret their associated label(s).
- **Prototype simplification**: Rather than projecting prototypes to complete sequences, use beam search to find a subsequence containing critical events. **NOTE**: the paper is ambiguous as to whether this is used during training (as the "projection" step) or whether this is just an interpretation tool.
