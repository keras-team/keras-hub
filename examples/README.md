# KerasNLP Example Models

This examples directory contains complete recipes for training popular model
architectures using KerasNLP. These are not part of the library itself, but
rather serve to demonstrate how to use the library for common tasks, while
simultaneously providing a mechanism to rigorously test library components.

This directory is complementary to the NLP examples on
[keras.io](https://keras.io/examples/). If you want to contribute a KerasNLP
example and you're not sure if it should live on keras.io or in this directory,
here's how they differ:

- If an example can fit in < 300 lines of code and run in a Colab,
  put it on keras.io.
- If an example is too big for a single script or has high compute requirements,
  add it here.

In general, we will have a fairly high bar for what models we support in this
directory. They should be widely used, practical models for solving standard
problems in NLP.

## Anatomy of an example

Given a model named `modelname`, which involves both pretraining and finetuning
on a downstream task, the contents of the `modelname` directory should be as
follows:

```shell
modelname
├── README.md
├── __init__.py
├── modelname_config.py
├── modelname_model.py
├── modelname_preprocess.py
├── modelname_train.py
└── modelname_finetune_X.py
```

- `README.md`: The README should contain complete instructions for downloading
  data and training a model from scratch.
- `__init__.py`: Empty (it's for imports).
- `modelname_config.py`: This file should contain most of the configuration for
  the model architecture, learning rate, etc, using simple Python constants. We
  would like to avoid complex configuration setups (json, yaml, etc).
- `modelname_preprocess.py`: If necessary. Standalone script to preprocess
  inputs. If possible, prefer doing preprocessing dynamically with tf.data
  inside the training and finetuning scripts.
- `modelname_model.py`: This file should contain the actual `keras.Model` and
  any custom layers needed for the example. Use KerasNLP components where ever
  possible.
- `modelname_train.py`: This file should be a runnable training script for
  pretraining. If possible, this script should preprocess data dynamically
  during training using `tf.data` and KerasNLP components (e.g. tokenizers).
- `modelname_finetune_X.py`: Optional. There can be any number of these files,
  for each task `X` we would like to support for finetuning. The file should be
  a runnable training script which loads and finetunes a pretrained model.

## Instructions for running on Google Cloud

TODO(https://github.com/keras-team/keras-nlp/issues/178)
