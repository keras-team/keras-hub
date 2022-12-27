# Style Guide

## Use `black`

For the most part, following our code style is very simple, we just use
[black](https://github.com/psf/black) to format code. See our
[Contributing Guide](CONTRIBUTING.md) for how to run our formatting scripts.

## Naming of Layers and Models

Capitalize all acronyms, e.g. LSTM not Lstm, KLDivergence not KlDivergence,
GPT2, XLMRoberta, etc.

Files should be named with snake case, and an acronym should be consider a
single "segment". For example XLMRoberta would map to xlm_roberta.py filename.

When a specific abbreviation is very common and is pronounceable (acronym),
consider it as a standalone word, e.g. Bert, Deberta, etc. In this case, "Bert"
is considered as a common noun and not an abbreviation anymore.

## Naming of Models and Presets

Naming of models and presets is a difficult and important element of our
library usability. In general we try to to follow the branding of "upstream"
model naming, subject to the consistency constraints laid out here.

- The model and preset names should be recognizable to users familiar with the
  original release. E.g. the model that goes with the "DeBERTaV3" paper should
  be called `DebertaV3`. A release of a [toxic-bert](https://huggingface.co/unitary/toxic-bert)
  checkpoint for `keras_nlp.models.Bert`, should include the string
  `"toxic_bert"`.
- All preset names should include the language of the pretraining data. If three
  or more language are supported, the preset name should include `"multi"` (not
  the single letter "m").
- If a preset lowercases input for cased-based languages, the preset name should
  be marked with `"uncased"`.
- Don't abbreviate size names. E.g. "xsmall" or "XL" in an original checkpoint
  releases should map to `"extra_small"` or `"extra_large"` in a preset names.
- No configuration in names. E.g. use "bert_base" instead of
  "bert_L-12_H-768_A-12".

When in doubt, readability should win out!

## File names

When possible, keep publicly documented classes in their own files, and make
the name of the class match the filename. E.g. the `BertClassifer` model should
be in `bert_classifier.py`, and the `TransformerEncoder` layer
should be in `transformer_encoder.py`

Small and/or unexported utility classes may live together along with code that
uses it if convenient, e.g., our `BytePairTokenizerCache` is collocated in the
same file as our `BytePairTokenizer`.

## Import keras and keras_nlp as top-level objects

Prefer importing `tf`, `keras` and `keras_nlp` as top-level objects. We want
it to be clear to a reader which symbols are from `keras_nlp` and which are
from core `keras`.

For guides and examples using KerasNLP, the import block should look as follows:

```python
import keras_nlp
import tensorflow as tf
from tensorflow import keras
```

❌ `tf.keras.activations.X`<br/>
✅ `keras.activations.X`

❌ `layers.X`<br/>
✅ `keras.layers.X` or `keras_nlp.layers.X`

❌ `Dense(1, activation='softmax')`<br/>
✅ `keras.layers.Dense(1, activation='softmax')`

For KerasNLP library code, `keras_nlp` will not be directly imported, but
`keras` should still be used as a top-level object used to access library
symbols.

## Ideal layer style

When writing a new KerasNLP layer (or tokenizer or metric), please make sure to
do the following:

- Accept `**kwargs` in `__init__` and forward this to the super class.
- Keep a python attribute on the layer for each `__init__` argument to the
  layer. The name and value should match the passed value.
- Write a `get_config()` which chains to super.
- Document the layer behavior thoroughly including call behavior though a
  class level docstring. Generally methods like `build()` and `call()` should
  not have their own docstring.
- Docstring text should start on the same line as the opening quotes and
  otherwise follow [PEP 257](https://peps.python.org/pep-0257/).
- Document the
  [masking](https://keras.io/guides/understanding_masking_and_padding/) behavior
  of the layer in the class level docstring as well.
- Always include usage examples using the full symbol location in `keras_nlp`.
- Include a reference citation if applicable.

````python
class PositionEmbedding(keras.layers.Layer):
    """A layer which learns a position embedding for input sequences.

    This class accepts a single dense tensor as input, and will output a
    learned position embedding of the same shape.

    This class assumes that in the input tensor, the last dimension corresponds
    to the features, and the dimension before the last corresponds to the
    sequence.

    This layer does not supporting masking, but can be combined with a
    `keras.layers.Embedding` for padding mask support.

    Args:
        sequence_length: The maximum length of the dynamic sequence.

    Examples:

    Direct call.
    >>> layer = keras_nlp.layers.PositionEmbedding(sequence_length=10)
    >>> layer(tf.zeros((8, 10, 16))).shape
    TensorShape([8, 10, 16])

    Combining with a token embedding.
    ```python
    seq_length = 50
    vocab_size = 5000
    embed_dim = 128
    inputs = keras.Input(shape=(seq_length,))
    token_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )(inputs)
    position_embeddings = keras_nlp.layers.PositionEmbedding(
        sequence_length=seq_length
    )(token_embeddings)
    outputs = token_embeddings + position_embeddings
    ```

    Reference:
     - [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
    """

    def __init__(
        self,
        sequence_length,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)

    def build(self, input_shape):
        super().build(input_shape)
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            "embeddings",
            shape=[self.sequence_length, feature_size],
        )

    def call(self, inputs):
        shape = tf.shape(inputs)
        input_length = shape[-2]
        position_embeddings = self.position_embeddings[:input_length, :]
        return tf.broadcast_to(position_embeddings, shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
            }
        )
        return config
````
