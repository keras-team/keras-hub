# KerasNLP: Modular NLP Workflows for Keras
[![](https://github.com/keras-team/keras-nlp/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-nlp/actions?query=workflow%3ATests+branch%3Amaster)
![Python](https://img.shields.io/badge/python-v3.9.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-nlp/issues)

KerasNLP is a natural language processing library that works natively
with TensorFlow, JAX, or PyTorch. Built on [multi-backend Keras](https://keras.io/keras_core/announcement/)
(Keras 3), these models, layers, metrics, and tokenizers can be trained and
serialized in any framework and re-used in another without costly migrations.

KerasNLP supports users through their entire development cycle. Our workflows
are built from modular components that have state-of-the-art preset weights when
used out-of-the-box and are easily customizable when more control is needed.

This library is an extension of the core Keras API; all high-level modules are
[`Layers`](https://keras.io/api/layers/) or
[`Models`](https://keras.io/api/models/) that receive that same level of polish
as core Keras. If you are familiar with Keras, congratulations! You already
understand most of KerasNLP.

See our [Getting Started guide](https://keras.io/guides/keras_nlp/getting_started)
to start learning our API. We welcome [contributions](CONTRIBUTING.md).

## Quick Links

### For everyone

- [Home Page](https://keras.io/keras_nlp)
- [Developer Guides](https://keras.io/guides/keras_nlp)
- [API Reference](https://keras.io/api/keras_nlp)
- [Getting Started guide](https://keras.io/guides/keras_nlp/getting_started) 

### For contributors

- [Contributing Guide](CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)
- [Style Guide](STYLE_GUIDE.md)
- [API Design Guide](API_DESIGN_GUIDE.md)
- [Call for Contributions](https://github.com/keras-team/keras-nlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22)

## Installation

To install the latest official release:

```
pip install keras-nlp --upgrade
```

To install the latest unreleased changes to the library, we recommend using
pip to install directly from the master branch on github:

```
pip install git+https://github.com/keras-team/keras-nlp.git --upgrade
```

## Quickstart

Fine-tune BERT on a small sentiment analysis task using the
[`keras_nlp.models`](https://keras.io/api/keras_nlp/models/) API:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow", or "torch".

import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
# Load a BERT model.
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased", 
    num_classes=2,
    activation="softmax",
)
# Fine-tune on IMDb movie reviews.
classifier.fit(imdb_train, validation_data=imdb_test)
# Predict two new examples.
classifier.predict(["What an amazing movie!", "A total waste of my time."])
```

For more in depth guides and examples, visit https://keras.io/keras_nlp/.

## Configuring your backend

**Keras 3** is an upcoming release of the Keras library which supports
TensorFlow, Jax or Torch as backends. This is supported today in KerasNLP,
but will not be enabled by default until the official release of Keras 3. If you
`pip install keras-nlp` and run a script or notebook without changes, you will
be using TensorFlow and **Keras 2**.

If you would like to enable a preview of the Keras 3 behavior, you can do
so by setting the `KERAS_BACKEND` environment variable. For example:

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

Until the Keras 3 release, KerasNLP will use a preview of Keras 3 on PyPI named
[keras-core](https://pypi.org/project/keras-core/).

> [!IMPORTANT]
> If you set `KERAS_BACKEND` variable, you should `import keras_core as keras`
> instead of `import keras`. This is a temporary step until Keras 3 is out!

To restore the default **Keras 2** behavior, `unset KERAS_BACKEND` before
importing Keras and KerasNLP.

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
BART, DeBERTa, DistilBERT, GPT-2, OPT, RoBERTa, Whisper, and XLM-RoBERTa.

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
