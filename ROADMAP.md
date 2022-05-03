# Roadmap

## What KerasNLP is

- **A high-quality library of modular building blocks.** KerasNLP components
  follow an established Keras interface (e.g. keras.layers.Layer,
  keras.metrics.Metric, or keras_nlp.tokenizers.Tokenizer), and make it easy to
  assemble state-of-the-art NLP workflows.

- **A collection of guides and examples.** This effort is split between two
  locations:
  
  On [keras.io](keras.io/keras_nlp), we host a collection of small-scale,
  easily accessible guides showing end-to-end workflows using KerasNLP.

  In this repository, we host a collection of
  [examples](https://github.com/keras-team/keras-nlp/tree/master/examples) on
  how to train large-scale, state-of-the-art models from scratch. This is not
  part of the library itself, but rather a way to vet our components and show
  best practices.

- **A community of NLP practioners.** KerasNLP is a actively growing project,
  and we welcome contributors on all fronts of our development. We hope that our
  guides and examples can be both a valuable resource to experienced
  practitioners, and a accessible entry point to newcomers to the field.

## What KerasNLP is not

- **KerasNLP is not a research library.** Researchers may use it, but we do not
  consider researchers to be our target audience. Our target audience is
  applied NLP engineers with experimentation and production needs. KerasNLP
  should make it possible to quickly re-implement industry-strength versions of
  the latest generation of architectures produced by researchers, but we don't
  expect the research effort itself to be built on top of KerasNLP. This enables
  us to focus on usability and API standardization, and produce objects that
  have a longer lifespan than the average research project.

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
  building, evaluation metrics, or visualization utils.

- **Be resolutely high-level.** Even if something is easy to do by hand in 5
  lines, package it as a one liner.

- **Balance ease of use and flexibility.** Simple things should be easy, and
  arbitrarily advanced use cases should be possible. There should always be a
  "we need to go deeper" path available to our most expert users.

- **Grow as a platform and as a community.** KerasNLP development should be
  driven by the community, with feature and release planning happening out in
  the open on GitHub.

## Areas of interest

At this point in our development cycle, we are primarily interested in providing
building blocks for a short list of "greatest hits" NLP models (such as BERT,
GPT-2, word2vec). Given a popular model architecture (e.g. a
sequence-to-sequence transformer like T5) and a end-to-end task (e.g.
summarization), we should have a clear code example in mind and a list of
components to use.

Note that while we will be supporting large-scale Transformer models as a
key offering from our library, but we are not a strictly Transformer-based
modeling library. We aim to support simple techniques such as n-gram models and
word2vec embeddings, and make it easy to hop between different approaches.

Current focus areas:

- In-graph tokenization leveraging
  [Tensorflow Text](https://www.tensorflow.org/text). We aim to have a fully
  featured offering of character, word, and sub-word tokenizers that run
  within the Tensorflow graph.
- Scalable and easily trainable modeling
  [examples](https://github.com/keras-team/keras-nlp/tree/master/examples)
  runnable on Google Cloud. We will continue to port our BERT example to run
  entirely on keras_nlp components for both training and preprocessing, and
  give easy recipes for running multi-worker training. Once this is done, we
  would like to extend this effort to other popular architectures.
- Text generation workflows. We would like to support text generation from
  trained models using greedy or beam search in a clear and easy to use
  workflow.
- Data augmentation preprocessing layers for domains with limited data. These
  layers will allow easily defining `tf.data` pipelines that augment input
  example sentences on the fly.
- Metrics for model evaluation, such a ROUGE and BLEU for evaluating translation
  quality.

## Citations

At this moment in time, we have no set citation bar for development, but due to
the newness of the library we want to focus our efforts on a small subset of the
best known and most effective NLP techniques.

Proposed components should usually either be part of a very well architecture
(think 1000s of citations) or contribute in some meaningful way to the usability
of an end-to-end workflow.

## Pre-trained modeling workflows

Pre-training many modern NLP models is prohibitively expensive and
time-consuming for an average user. A key goal with the KerasNLP project is to
support easy use of pre-trained models using KerasNLP components.

We are working with the rest of the Tensorflow ecosystem (e.g. TF Hub), to
provide a coherent plan for accessing pre-trained models. We will continue to
share updates as they are available.
