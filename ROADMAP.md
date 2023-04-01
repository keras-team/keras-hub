# Roadmap

## What KerasNLP is

- **A high-quality library of modular building blocks.** KerasNLP components
  follow an established Keras interface (e.g. `keras.layers.Layer`,
  `keras.metrics.Metric`, or `keras_nlp.tokenizers.Tokenizer`), and make it easy
  to assemble state-of-the-art NLP workflows.

- **A collection of guides and examples.** This effort is split between two
  locations. On [keras.io](keras.io/keras_nlp), we host a collection of
  small-scale, easily accessible guides showing end-to-end workflows using
  KerasNLP. In this repository, we host a collection of
  [examples](https://github.com/keras-team/keras-nlp/tree/master/examples) on
  how to train large-scale, state-of-the-art models from scratch. This is not
  part of the library itself, but rather a way to vet our components and show
  best practices.

- **A community of NLP practitioners.** KerasNLP is an actively growing project,
  and we welcome contributors on all fronts of our development. We hope that our
  guides and examples can be both a valuable resource to experienced
  practitioners and an accessible entry point to newcomers to the field.

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

- **KerasNLP is not a repository of low-level string ops, like `tf.text`.**
  KerasNLP is fundamentally an extension of the Keras API: it hosts Keras
  objects, like layers, metrics, or callbacks. Low-level C++ ops should go
  directly to [Tensorflow Text](https://www.tensorflow.org/text) or
  core Tensorflow.

- **KerasNLP is not a Transformer only library.**
  Transformer based models are a key offering for KerasNLP, and they should be
  easy to train and use within the library. However, we need to support other
  types of models, such as n-gram or word2vec approaches that might run more
  easily on limited hardware. We will always want the most practical tool for
  the task, regardless of the architecture.

## Focus areas for 2022

At this point in our development cycle, we are primarily interested in providing
building blocks for a short list of "greatest hits" NLP models (such as BERT,
GPT-2, word2vec). Given a popular model architecture (e.g. a
sequence-to-sequence transformer like T5) and an end-to-end task (e.g.
summarization), we should have a clear code example in mind and a list of
components to use.

Below, we describe our areas of focus for the year in more detail.

### Easy-to-use and feature-complete tokenization

KerasNLP should be the "go-to" tokenization solution for Keras model training
and deployment by the end of 2022.

The major tasks within this effort:

- Work with Tensorflow Text to continue to support a growing range of
  tokenization options and popular vocabulary formats. For example, we would
  like to add support for byte-level BPE tokenization (the RoBERTa and GPT
  tokenizer) within the Tensorflow graph.
- Pre-trained sub-word tokenizers for any language. Training a tokenizer can
  add a lot of friction to a project, particularly when working in a
  language where examples are less readily available. We would like to support
  a pre-trained tokenization offering that allows a user to choose a tokenizer,
  language, and vocabulary size and then download an off the shelf vocabulary.
- A standardized way to train tokenizer vocabularies. As another way to
  reduce the friction of training a tokenizer, we should offer a standardized
  experience for training new vocabularies.

### Scalable examples of popular model architectures using KerasNLP

We would like our
[examples](https://github.com/keras-team/keras-nlp/tree/master/examples)
directory to contain scalable implementations of popular model
architectures easily runnable on Google Cloud. Note that these will not be
shipped with the library itself.

These examples will serve two purposes—a demonstration to the community of how
models can be built using KerasNLP, and a way to vet our the performance and
accuracy of our library components on both TPUs and GPUs at scale.

At this moment in time, our focus is on polishing our BERT example. We would
like it to run entirely on KerasNLP components for both training and
preprocessing, and come with easy recipes for running multi-worker training
jobs. Once this is done, we would like to extend our examples directory to other
popular architectures (e.g. RoBERTa and ELECTRA).

As we move forward with KerasNLP as a whole, we expect development for new
components (say, a new attention mechanism) to happen in tandem with an
example demonstrating the component in an end-to-end architecture.

By the end of 2022, we should have an actively growing collection of examples
models, with a standardized set of training scripts, that match expected
performance as reported in publications.

On the scalability front, we should have at least one example demonstrating both
data parallel and model parallel training, in a multi-worker GPU and TPU
setting, leveraging
[DTensor](https://www.tensorflow.org/guide/dtensor_overview) for distributed
support.

### Tools for data preprocessing and postprocessing for end-to-end workflows

It should be easy to take a trained Keras language model and use it for a wide
range of real world NLP tasks. We should support classification, text
generation, summarization, translation, name-entity recognition, and question
answering. We should have a guide for each of these tasks using KerasNLP by
the end of 2022.

We are looking to develop simple, modular components that make it easy to build
end-to-end workflows for each of these tasks.

Currently, projects in this area include:

- Utilities for generating sequences of text using greedy or beam search.
- Metrics for evaluating the quality of generated sequences, such a ROUGE and
  BLEU.
- Data augmentation preprocessing layers for domains with limited data. These
  layers will allow easily defining `tf.data` pipelines that augment input
  example sentences on the fly.

### Accessible guides and examples on keras.io

For all of the above focus areas, we would like to make ensure we have an
industry leading collection of easy to use guides and examples.

These examples should be easy to follow, run within a colab notebook, and
provide a practical starting place for solving most real-world NLP problems.
Given the scale of modern NLP models, this will often involve scaling down the
model or data size for a particular task while preserving the core of what we
are trying to explain to the reader.

This will continue to be a key investment area for the library. If you have an
idea for a guide or example, please open an issue to discuss.

By the end of 2022, most new NLP examples on keras.io should use
KerasNLP library.

## Citation bar

At this moment in time, we have no set citation bar for development, but due to
the newness of the library we want to focus our efforts on a small subset of the
best known and most effective NLP techniques.

Proposed components should usually either be part of a very well known
architecture or contribute in some meaningful way to the usability of an
end-to-end workflow.

## Pretrained modeling workflows

Pretraining many modern NLP models is prohibitively expensive and
time-consuming for an average user. A key goal with for the KerasNLP project is
to have KerasNLP components available in a pretrained model offering of some
form.

We are working with the rest of the Tensorflow ecosystem, to provide a coherent
plan for accessing pretrained models. We will continue to share updates as they
are available.
