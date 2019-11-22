# TensorFlow-ProSeNet

This is a `tf.keras` implementation of [Interpretable and Steerable Sequence Learning via Prototypes](https://arxiv.org/abs/1907.09728) (Ming et al., 2019). It's unlikely I'll implement a mechanism for the "steering" part, but most of the interpretive stuff should be useful.

**Contributions are welcome!**

## Status

I have an implementation that *sort of* seems to work.

I'm currently testing on the [MIT-BIH Arrhythmia ECG Dataset](https://physionet.org/content/mitdb/1.0.0/), which is available the pre-processed format described in [Kachuee et al., 2018](https://arxiv.org/abs/1805.00794) from Kaggle: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/shayanfazeli/heartbeat/data#).

See [notebooks/test_arrythmia.ipynb](notebooks/test_arrythmia.ipynb). There are some outstanding issues...

No matter how I weight the different regularization terms, prototypes seems to collapse to a matrix of `ones` (or at least to a similarly uniform matrix), with the classifier weights tending towards vectors of `[1., 0., 0., 0., 0.]`. (The first class -- "Normal" beats -- accounts for ~83% of the dataset).

I've tried:
- Using `class_weights`, even minimizing the weight of the first class to be negligibly small. **Does not seem to help.**
- Training just the LSTM `encoder` first (achieves up to ~95% accuracy), then freezing those layers and training only the `prototypes_layer` and `classifier`. **Does not seem to help.**
- Heavier regularization (of both prototype vector diversity and classifier weights). **Does not seem to help.**

Some of these attempts at solutions have the effect of decreasing the imbalance in predicted class probabilities (e.g., the model outputs something like `[0.24, 0.19, 0.19, ...]` for every input sequence, rather than something like `[0.96, 0.01, 0.01, ...]`), but none of them seem to address the root problem.

Other ideas to try:
- Heavily downsampling the first class.
- Verifying custom regularization terms more rigorously.

## Additional Features

**Prototype Projection** has been implemented as a `Callback`, but I have not yet implemented **Prototype Simplification** (via beam search). **NOTE**: the paper is somewhat ambiguous as to whether this is used during training (as the "projection" step) or whether this is just an interpretation tool. I'm pretty sure it's the latter, but I may raise an issue on [their template repo](https://github.com/myaooo/ProSeNet) just to be certain.

Additional functions to help with prototype interpretation will be helpful, but I want to make sure I can get the network to train properly first.
