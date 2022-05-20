# KerasNLP
[![](https://github.com/keras-team/keras-nlp/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-nlp/actions?query=workflow%3ATests+branch%3Amaster)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.5.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-nlp/issues)

KerasNLP is a simple and powerful API for building Natural Language Processing
(NLP) models within the Keras ecosystem.

KerasNLP provides modular building blocks following
standard Keras interfaces (layers, metrics) that allow you to quickly and
flexibly iterate on your task. Engineers working in applied NLP can leverage the
library to assemble training and inference pipelines that are both
state-of-the-art and production-grade.

KerasNLP can be understood as a horizontal extension of the Keras API —
components are first-party Keras objects that are too specialized to be
added to core Keras, but that receive the same level of polish as the rest of
the Keras API.

We are a new and growing project, and welcome [contributions](CONTRIBUTING.md).

## Quick Links

- [Home Page](https://keras.io/keras_nlp)
- [Developer Guides](https://keras.io/guides/keras_nlp)
- [API Reference](https://keras.io/api/keras_nlp)
- [Contributing Guide](CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)
- [Style Guide](STYLE_GUIDE.md)
- [API Design Guide](API_DESIGN_GUIDE.md)
- [Call for Contributions](https://github.com/keras-team/keras-nlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22)

## Quick Start

Install the latest release:

```
pip install keras-nlp --upgrade
```

Tokenize text, build a tiny transformer, and train a single batch:

```python
import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Tokenize some inputs with a binary label.
vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
sentences = ["The quick brown fox jumped.", "The fox slept."]
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=10,
)
x, y = tokenizer(sentences), tf.constant([1, 0])

# Create a tiny transformer.
inputs = keras.Input(shape=(None,), dtype="int32")
outputs = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=len(vocab),
    sequence_length=10,
    embedding_dim=16,
)(inputs)
outputs = keras_nlp.layers.TransformerEncoder(
    num_heads=4,
    intermediate_dim=32,
)(outputs)
outputs = keras.layers.GlobalAveragePooling1D()(outputs)
outputs = keras.layers.Dense(1, activation="sigmoid")(outputs)
model = keras.Model(inputs, outputs)

# Run a single batch of gradient descent.
model.compile(loss="binary_crossentropy", jit_compile=True)
model.train_on_batch(x, y)
```

## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to
provide backwards compatibility guarantees both for code and saved models built
with our components. While we continue with pre-release `0.y.z` development, we
may break compatibility at any time and APIs should not be consider stable.

## Citing KerasNLP

If KerasNLP helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerasnlp2022,
  title={KerasNLP},
  author={Watson, Matthew, and Qian, Chen, and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-nlp}},
}
```

Thank you to all of our wonderful contributors!

<a href="https://github.com/keras-team/keras-nlp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=keras-team/keras-nlp" />
</a>