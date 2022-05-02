# Roadmap

## What KerasNLP is

KerasNLP is focused on a few core offerings:

- A high-quality library of modular building blocks for modern NLP workflows.
- A collection of guides and examples on [keras.io](keras.io/keras_nlp) showing
  how to use these components to solve end-to-end NLP tasks.
- A collection of examples in this repository, showing how to use these
  components at scale to train state-of-art models from scratch. This is not
  part of the library itself, but rather a way to vet our components and model
  best practices.

Contributions on any of these fronts are welcome!

## What KerasNLP is not

- **KerasNLP is not a research library.** Researchers may use it, but we do not
  consider researchers to be our target audience. Our target audience is
  applied NLP engineers with experimentation and production needs. KerasNLP
  should make it possible to quickly reimplement industry-strength versions of
  the latest generation of architectures produced by researchers, but we don't
  expect the research
  effort itself to be built on top of KerasNLP. This enables us to focus on
  usability and API standardization, and produce objects that have a longer
  lifespan than the average research project.

- **KerasNLP is not a repository of blackbox end-to-end solutions.**

    KerasNLP is focused on modular and reusable building blocks. In the process
    of developing these building blocks, we will by necessity implement
    end-to-end workflows, but they're intended purely for demonstration and
    grounding purposes, they're not our main deliverable.

- **KerasNLP is not a repository of low-level string ops, like tf.text.**

    KerasNLP is fundamentally an extension of the Keras API: it hosts Keras
    objects, like layers, metrics, or callbacks. Low-level C++ ops should go
    directly to [Tensorflow Text](https://www.tensorflow.org/text) or
    core Tensorflow.

## Philosophy

- **Let user needs be our compass.** Any modular building block that NLP
  practitioners need is in scope, whether it's data loading, augmentation, model
  building, evaluation metrics, visualization utils...
- **Be resolutely high-level.** Even if something is easy to do by hand in 5
  lines, package it as a one liner.
- **Balance ease of use and flexibility** – simple things should be easy, and
  arbitrarily advanced use cases should be possible. There should always be a
  "we need to go deeper" path available to our most expert users.
- **Grow as a platform and as a community** – KerasNLP development should 

## Areas of interest

At this point in our development cycle, we are primarily interested in providing
building blocks for a short list of "greatest hits" NLP models (e.g. BERT,
GPT-2, word2vec).

We are focusing on components that follow an established Keras interface
(e.g. keras.layers.Layer, keras.metrics.Metric, or
keras_nlp.tokenizers.Tokenizer).

Note that while we will be supporting large-scale, pre-trained Transformer as a
key offering from our library, but we are not a strictly Transformer-based
modeling library. We aim to support simple techniques such as n-gram models and
word2vec embeddings, and make it easy to hop between different approaches.

## Pre-trained modeling workflows


