# KerasNLP
[![](https://github.com/keras-team/keras-nlp/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-nlp/actions?query=workflow%3ATests+branch%3Amaster)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.5.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-nlp/issues)

KerasNLP is a repository of modular building blocks (layers, metrics, losses).
Engineers working with applied natural language processing can leverage it to
rapidly assemble training and inference pipelines that are both state-of-the-art
and production-grade. Common use cases for application include sentiment
analysis, named entity recognition, text generation, etc.

KerasNLP can be understood as a horizontal extension of the Keras API: they're
new first-party Keras objects (layers, metrics, etc) that are too specialized to
be added to core Keras, but that receive the same level of polish and backwards
compatibility guarantees as the rest of the Keras API and that are maintained by
the Keras team itself (unlike TFAddons).

Currently, KerasNLP is operating pre-release. Upon launch of KerasNLP 1.0, full
API docs and code examples will be available.

## Contributors

If you'd like to contribute, please see our [contributing guide](CONTRIBUTING.md).

The fastest way to find a place to contribute is to browse our
[open issues](https://github.com/keras-team/keras-nlp/issues) and find an
unclaimed issue to work on. We tag good introductory issues with the
`good first issue` label.

If you would like to propose a new symbol or feature, please open an issue to
discuss. Be aware the design for new features may take longer at this early
stage. If you have a design in mind, please include a colab notebook showing the
proposed design in a end-to-end example.

## Roadmap

This is an early stage project, and we are actively working on a more detailed
roadmap to share soon. For now, most of our immediate planning is done through
GitHub issues.

At this stage, we are primarily building components for a short list of
"greatest hits" NLP models (e.g. BERT, GPT-2, word2vec). We will be focusing
on components that follow a established Keras interface (e.g.
`keras.layers.Layer`, `keras.metrics.Metric`, or
`keras_nlp.tokenizers.Tokenizer`).

As we progress further with the library, we will attempt to cover an ever
expanding list of widely cited model architectures.

## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to
provide backwards compatibility guarantees both for code and saved models built
with our components. While we continue with pre-release `0.y.z` development, we
may break compatibility at any time and APIs should not be consider stable.

Thank you to all of our wonderful contributors!

<a href="https://github.com/keras-team/keras-nlp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=keras-team/keras-nlp" />
</a>
