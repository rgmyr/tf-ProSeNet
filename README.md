# TensorFlow-ProSeNet

This is a `tf.keras` implementation of [Interpretable and Steerable Sequence Learning via Prototypes](https://arxiv.org/abs/1907.09728) (Ming et al., 2019). It's unlikely I'll implement a mechanism for the "steering" part, but most of the interpretive stuff should be useful to me.

**Contributions are welcome!**

## Status

The implementation needs troubleshooting / debugging work.

## MIT-BIH Dataset

I'm currently testing on the [MIT-BIH Arrhythmia ECG Dataset](https://physionet.org/content/mitdb/1.0.0/), which is available the pre-processed format described in ([Kachuee et al., 2018](https://arxiv.org/abs/1805.00794)) on Kaggle Datasets: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/shayanfazeli/heartbeat/data#).

See [notebooks/test_arrythmia.ipynb](notebooks/test_arrythmia.ipynb). There are some outstanding issues...

No matter how I weight the different regularization terms, the classifier weights tend towards vectors of `[+const, 0., 0., 0., 0.]`. (The first class -- "Normal" beats -- accounts for ~83% of the dataset).

I've tried:
- Using `class_weights`, even minimizing the weight of the first class to be negligibly small. **Does not seem to help.**
- Training just the LSTM `encoder` first (achieves up to ~95% accuracy), then freezing those layers and training only the `prototypes_layer` and `classifier`. **Does not seem to help.**
- Heavier regularization (of both prototype vector diversity and classifier weights). **Does not seem to help.**

Other ideas to try:
- Heavily downsampling the first class.
- Verifying custom regularization terms more rigorously.
- Put cross entropy loss on logits (rather than softmax)

## Synthetic Signals Dataset

I've written a `SyntheticSignalsDataset` class that generates saw/square/sine signals. This is a very simple dataset which a simple LSTM can easily master. I'm using this dataset to troubleshoot. See [notebooks/test_synthetic.ipynb](notebooks/test_synthetic.ipynb).

This dataset seems to work OK after I fixed a bug in the diversity regularization function -- still troubleshooting the ECG one.

## Additional Features

**Prototype Projection** has been implemented as a `Callback`, but I have not yet implemented **Prototype Simplification** (via beam search). **NOTE**: the paper is somewhat ambiguous as to whether this is used during training (as the "projection" step) or whether this is just an interpretation tool. I'm pretty sure it's the latter, but I may raise an issue on [their template repo](https://github.com/myaooo/ProSeNet) just to be certain.

Additional functions to help with prototype interpretation will be helpful, but I want to make sure I can get the network to train properly first.
