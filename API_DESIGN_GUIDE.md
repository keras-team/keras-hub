# API Design Guide

Before reading this document, please read the
[Keras API design guidelines](https://github.com/keras-team/governance/blob/master/keras_api_design_guidelines.md).

Below are some design considerations specific to KerasNLP.

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
  driven by the community, with feature and release planning happening in
  the open on GitHub.

## Avoid new dependencies

The core dependencies of KerasNLP are Keras, NumPy, TensorFlow, and
[Tensorflow Text](https://www.tensorflow.org/text).

We strive to keep KerasNLP as self-contained as possible, and avoid adding
dependencies to projects (for example NLTK or spaCy) for text preprocessing.

In rare cases, particularly with tokenizers and metrics, we may need to add
an external dependency for compatibility with the "canonical" implementation
of a certain technique. In these cases, avoid adding a new package dependency,
and add installation instructions for the specific symbol:

```python
try:
    import rouge_score
except ImportError:
    rouge_score = None

class Rouge(keras.metrics.Metric):
    def __init__(self):
        if rouge_score is None:
            raise ImportError(
                "ROUGE metric requires the `rouge_score` package."
                "Please install it with `pip install rouge_score`."
            )
```

Additionally, to ensure that unit tests don't fail, please add the corresponding
library to the `extras_require["tests"]` list in `setup.py`.

## Keep computation inside TensorFlow graph

Our layers, metrics, and tokenizers should be fast and efficient, which means
running inside the
[TensorFlow graph](https://www.tensorflow.org/guide/intro_to_graphs)
whenever possible. This means you should be able to wrap annotate a function
calling a layer, metric or loss with `@tf.function` without running into issues.

[tf.strings](https://www.tensorflow.org/api_docs/python/tf/strings) and
[tf.text](https://www.tensorflow.org/text/api_docs/python/text) provides a large
surface on TensorFlow operations that manipulate strings. If an low-level (c++)
operation we need is missing, we should add it in collaboration with core
TensorFlow or TensorFlow Text. KerasNLP is a python-only library.

We should also strive to keep computation XLA compilable wherever possible (e.g.
`tf.function(jit_compile=True)`). For trainable modeling components this is
particularly important due to the performance gains offered by XLA. For
preprocessing and postprocessing, XLA compilation is not a requirement.

## Support tf.data for text preprocessing and augmentation

In general, our preprocessing tools should be runnable inside a
[tf.data](https://www.tensorflow.org/guide/data) pipeline, and any augmentation
to training data should be dynamic (runnable on the fly during training rather
than precomputed).

We should design our preprocessing workflows with tf.data in mind, and support
both batched and unbatched data as input to preprocessing layers.

## Prioritize multi-lingual support

We strive to keep KerasNLP a friendly and useful library for speakers of all
languages. In general, prefer designing workflows that are language agnostic,
and do not involve logic (e.g. stemming) that need to be rewritten
per-language.

It is OK for new workflows to not come with of the box support for all
languages in a first release, but a design that does not include a plan for
multi-lingual support will be rejected.
