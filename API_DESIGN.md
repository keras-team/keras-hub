# KerasNLP Design Guidelines

KerasNLP uses the same API design guidelines as the rest of the Keras
ecosystem, documented [here]
https://github.com/keras-team/governance/blob/master/keras_api_design_guidelines.md).
Anyone hoping to contribute to KerasNLP API design is strongly encouraged to
read through that document in it's entirety.

Below are some design considerations specific to KerasNLP.

## Dependencies

The core dependencies of KerasNLP are Numpy, TensorFlow, Keras, and
[Tensorflow Text](https://www.tensorflow.org/text).

We strive to keep KerasNLP as self-contained as possible, and will not add
dependencies to projects like NLTK or spaCy for text preprocessing.

In rare cases, particularly with tokenizers and metrics, we may need to add
an external dependency for compatibility with the "canonical" implementation
of a certain technique. In these cases, avoid adding a new package dependency,
and add installation instructions for the specific symbol:

```python
try:
    import rouge_score
except ImportError:
    pass

class RougeL(keras.metrics.Metric):
    def __init__(self):
        if rouge_score is None:
            raise ImportError(
                'RougeL metrics requires the rouge_score package. '
                '`pip install rouge-score`.')
```

## TensorFlow graph support

Our layers, metrics, and tokenizers should be fast and efficient, which means
running inside the
[TensorFlow graph](https://www.tensorflow.org/guide/intro_to_graphs)
whenever possible.

[tf.strings](https://www.tensorflow.org/api_docs/python/tf/strings) and
[tf.text](https://www.tensorflow.org/text/api_docs/python/text) provides a large
surface on TensorFlow operations that manipulate strings.

If an low-level (c++) operation we need is missing, we should add it in
collaboration with core TensorFlow or TensorFlow Text. KerasNLP is a python-only
library.

## Multi-lingual support

We strive to keep KerasNLP a friendly and useful library for speakers of all
languages. In general, prefer designing workflows that are language agnostic,
and do not involve details (e.g. stemming) that need to be rewritten
per-language.

It is OK for new workflows to not come with of the box support for all
languages in a first release, but a design that does not include a plan for
multi-lingual support will be rejected.
