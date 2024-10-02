# KerasHub: Multi-framework Models
[![](https://github.com/keras-team/keras-hub/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-hub/actions?query=workflow%3ATests+branch%3Amaster)
![Python](https://img.shields.io/badge/python-v3.9.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-hub/issues)

> [!IMPORTANT]
> 📢 KerasNLP is becoming KerasHub! 📢 Read
> [the announcement](https://github.com/keras-team/keras-hub/issues/1831).
>
> We have renamed the repo to KerasHub in preparation for the release, but have not yet
> released the new package. Follow the announcement for news.

KerasHub is a library that supports natural language processing, computer
vision, audio, and multimodal backbones and task models, working natively with
TensorFlow, JAX, or PyTorch. KerasHub provides a repository of pre-trained
models and a collection of lower-level building blocks for these tasks. Built
on Keras 3, models can be trained and serialized in any framework and re-used
in another without costly migrations.

This library is an extension of the core Keras API; all high-level modules are
Layers and Models that receive that same level of polish as core Keras.
If you are familiar with Keras, congratulations! You already understand most of
KerasHub.

All models support JAX, TensorFlow, and PyTorch from a single model
definition and can be fine-tuned on GPUs and TPUs out of the box. Models can
be trained on individual accelerators with built-in PEFT techniques, or
fine-tuned at scale with model and data parallel training. See our
[Getting Started guide](https://keras.io/guides/keras_nlp/getting_started)
to start learning our API. Browse our models on
[Kaggle](https://www.kaggle.com/organizations/keras/models).
We welcome contributions.

## Quick Links

### For everyone

- [Home Page](https://keras.io/keras_nlp)
- [Developer Guides](https://keras.io/guides/keras_nlp)
- [API Reference](https://keras.io/api/keras_nlp)
- [Pre-trained Models](https://www.kaggle.com/organizations/keras/models)

### For contributors

- [Contributing Guide](CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)
- [Style Guide](STYLE_GUIDE.md)
- [API Design Guide](API_DESIGN_GUIDE.md)
- [Call for Contributions](https://github.com/keras-team/keras-hub/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22)

## Quickstart

Fine-tune a BERT classifier on IMDb movie reviews:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch"!

import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)

# Load a BERT model.
classifier = keras_nlp.models.Classifier.from_preset(
    "bert_base_en",
    num_classes=2,
    activation="softmax",
)

# Fine-tune on IMDb movie reviews.
classifier.fit(imdb_train, validation_data=imdb_test)
# Predict two new examples.
classifier.predict(["What an amazing movie!", "A total waste of my time."])
```

Try it out [in a colab](https://colab.research.google.com/gist/mattdangerw/e457e42d5ea827110c8d5cb4eb9d9a07/kerasnlp-quickstart.ipynb).
For more in depth guides and examples, visit
[keras.io/keras_nlp](https://keras.io/keras_nlp/).

## Installation

KerasHub is currently in pre-release. Note that pre-release versions may
introduce breaking changes to the API in future versions. For a stable and
supported experience, we recommend installing `keras-nlp` version 0.15.1:

```bash
pip install keras-nlp==0.15.1
```

To try out the latest pre-release version of KerasHub, you can use
our nightly package:

```bash
pip install keras-hub-nightly
```

KerasHub currently requires TensorFlow to be installed for use of the
`tf.data` API for preprocessing. Even when pre-processing with `tf.data`,
training can still happen on any backend.

Read [Getting started with Keras](https://keras.io/getting_started/) for more
information on installing Keras 3 and compatibility with different frameworks.

> [!IMPORTANT]
> We recommend using KerasHub with TensorFlow 2.16 or later, as TF 2.16 packages
> Keras 3 by default.

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
