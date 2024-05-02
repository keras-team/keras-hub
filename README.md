# KerasNLP: Multi-framework NLP Models
[![](https://github.com/keras-team/keras-nlp/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-nlp/actions?query=workflow%3ATests+branch%3Amaster)
![Python](https://img.shields.io/badge/python-v3.9.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-nlp/issues)

KerasNLP is a natural language processing library that works natively
with TensorFlow, JAX, or PyTorch. KerasNLP provides a repository of pre-trained
models and a collection of lower-level building blocks for language modeling.
Built on Keras 3, models can be trained and serialized in any framework
and re-used in another without costly migrations.

This library is an extension of the core Keras API; all high-level modules are
Layers and Models that receive that same level of polish as core Keras.
If you are familiar with Keras, congratulations! You already understand most of
KerasNLP.

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
- [Call for Contributions](https://github.com/keras-team/keras-nlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22)

## Quickstart

Fine-tune BERT on IMDb movie reviews:

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

To install the latest KerasNLP release with Keras 3, simply run:

```
pip install --upgrade keras-nlp
```

To install the latest nightly changes for both KerasNLP and Keras, you can use
our nightly package.

```
pip install --upgrade keras-nlp-nightly
```

Note that currently, installing KerasNLP will always pull in TensorFlow for use
of the `tf.data` API for preprocessing. Even when pre-processing with `tf.data`,
training can still happen on any backend.

Read [Getting started with Keras](https://keras.io/getting_started/) for more
information on installing Keras 3 and compatibility with different frameworks.

> [!IMPORTANT]
> We recommend using KerasNLP with TensorFlow 2.16 or later, as TF 2.16 packages
> Keras 3 by default.

## Configuring your backend

If you have Keras 3 installed in your environment (see installation above),
you can use KerasNLP with any of JAX, TensorFlow and PyTorch. To do so, set the
`KERAS_BACKEND` environment variable. For example:

```shell
export KERAS_BACKEND=jax
```

Or in Colab, with:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_nlp
```

> [!IMPORTANT]
> Make sure to set the `KERAS_BACKEND` before import any Keras libraries, it
> will be used to set up Keras when it is first imported.

## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to
provide backwards compatibility guarantees both for code and saved models built
with our components. While we continue with pre-release `0.y.z` development, we
may break compatibility at any time and APIs should not be consider stable.

## Disclaimer

KerasNLP provides access to pre-trained models via the `keras_nlp.models` API.
These pre-trained models are provided on an "as is" basis, without warranties
or conditions of any kind. The following underlying models are provided by third
parties, and subject to separate licenses:
BART, BLOOM, DeBERTa, DistilBERT, GPT-2, Llama, Mistral, OPT, RoBERTa, Whisper,
and XLM-RoBERTa.

## Citing KerasNLP

If KerasNLP helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerasnlp2022,
  title={KerasNLP},
  author={Watson, Matthew, and Qian, Chen, and Bischof, Jonathan and Chollet, 
  Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-nlp}},
}
```

## Acknowledgements

Thank you to all of our wonderful contributors!

<a href="https://github.com/keras-team/keras-nlp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=keras-team/keras-nlp" />
</a>
