# KerasHub: Multi-framework Pretrained Models
[![](https://github.com/keras-team/keras-hub/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-hub/actions?query=workflow%3ATests+branch%3Amaster)
![Python](https://img.shields.io/badge/python-v3.9.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-hub/issues)

> [!IMPORTANT]
> 📢 KerasNLP is now KerasHub! 📢 Read
> [the announcement](https://github.com/keras-team/keras-hub/issues/1831).

**KerasHub** is a pretrained modeling library that aims to be simple, flexible,
and fast. The library provides [Keras 3](https://keras.io/keras_3/)
implementations of popular model architectures, paired with a collection of
pretrained checkpoints available on [Kaggle Models](https://kaggle.com/models/).
Models can be used with text, image, and audio data for generation, classification,
and many other built in tasks.

KerasHub is an extension of the core Keras API; KerasHub components are provided
as `Layer` and `Model` implementations. If you are  familiar with Keras,
congratulations! You already understand most of KerasHub.

All models support JAX, TensorFlow, and PyTorch from a single model
definition and can be fine-tuned on GPUs and TPUs out of the box. Models can
be trained on individual accelerators with built-in PEFT techniques, or
fine-tuned at scale with model and data parallel training. See our
[Getting Started guide](https://keras.io/guides/keras_hub/getting_started)
to start learning our API.

## Quick Links

### For everyone

- [Home page](https://keras.io/keras_hub)
- [Getting started](https://keras.io/keras_hub/getting_started)
- [Guides](https://keras.io/keras_hub/guides)
- [API documentation](https://keras.io/keras_hub/api)
- [Pre-trained models](https://keras.io/keras_hub/presets/)

### For contributors

- [Call for Contributions](https://github.com/keras-team/keras-hub/issues/1835)
- [Roadmap](https://github.com/keras-team/keras-hub/issues/1836)
- [Contributing Guide](CONTRIBUTING.md)
- [Style Guide](STYLE_GUIDE.md)
- [API Design Guide](API_DESIGN_GUIDE.md)

## Quickstart

Choose a backend:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch"!
```

Import KerasHub and other libraries:

```python
import keras
import keras_hub
import numpy as np
import tensorflow_datasets as tfds
```

Load a resnet model and use it to predict a label for an image:

```python
classifier = keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet",
    activation="softmax",
)
url = "https://upload.wikimedia.org/wikipedia/commons/a/aa/California_quail.jpg"
path = keras.utils.get_file(origin=url)
image = keras.utils.load_img(path)
preds = classifier.predict(np.array([image]))
print(keras_hub.utils.decode_imagenet_predictions(preds))
```

Load a Bert model and fine-tune it on IMDb movie reviews:

```python
classifier = keras_hub.models.TextClassifier.from_preset(
    "bert_base_en_uncased",
    activation="softmax",
    num_classes=2,
)
imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
classifier.fit(imdb_train, validation_data=imdb_test)
preds = classifier.predict(["What an amazing movie!", "A total waste of time."])
print(preds)
```

## Installation

To install the latest KerasHub release with Keras 3, simply run:

```
pip install --upgrade keras-hub
```

Our text tokenizers are based on TensorFlow Text. Hence, if you are using any
model which has language as a modality, you will have to run:

```
pip install --upgrade keras-hub[nlp]
```

To install the latest nightly changes for both KerasHub and Keras, you can use
our nightly package.

```
pip install --upgrade keras-hub-nightly
```

Currently, installing KerasHub will always pull in TensorFlow for use of the
`tf.data` API for preprocessing. When pre-processing with `tf.data`, training
can still happen on any backend.

Visit the [core Keras getting started page](https://keras.io/getting_started/)
for more information on installing Keras 3, accelerator support, and
compatibility with different frameworks.

## Configuring your backend

If you have Keras 3 installed in your environment (see installation above),
you can use KerasHub with any of JAX, TensorFlow and PyTorch. To do so, set the
`KERAS_BACKEND` environment variable. For example:

```shell
export KERAS_BACKEND=jax
```

Or in Colab, with:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_hub
```

> [!IMPORTANT]
> Make sure to set the `KERAS_BACKEND` **before** importing any Keras libraries;
> it will be used to set up Keras when it is first imported.

## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to
provide backwards compatibility guarantees both for code and saved models built
with our components. While we continue with pre-release `0.y.z` development, we
may break compatibility at any time and APIs should not be considered stable.

## Disclaimer

KerasHub provides access to pre-trained models via the `keras_hub.models` API.
These pre-trained models are provided on an "as is" basis, without warranties
or conditions of any kind. The following underlying models are provided by third
parties, and subject to separate licenses:
BART, BLOOM, DeBERTa, DistilBERT, GPT-2, Llama, Mistral, OPT, RoBERTa, Whisper,
and XLM-RoBERTa.

## Citing KerasHub

If KerasHub helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerashub2024,
  title={KerasHub},
  author={Watson, Matthew, and  Chollet, Fran\c{c}ois and Sreepathihalli,
  Divyashree, and Saadat, Samaneh and Sampath, Ramesh, and Rasskin, Gabriel and
  and Zhu, Scott and Singh, Varun and Wood, Luke and Tan, Zhenyu and Stenbit,
  Ian and Qian, Chen, and Bischof, Jonathan and others},
  year={2024},
  howpublished={\url{https://github.com/keras-team/keras-hub}},
}
```

## Acknowledgements

Thank you to all of our wonderful contributors!

<a href="https://github.com/keras-team/keras-hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=keras-team/keras-hub" />
</a>
