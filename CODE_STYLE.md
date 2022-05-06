# Style Guide

## Use black

For the most part, following are style guide is very simple, we just use
[black](https://github.com/psf/black) to format code. See our
[Contributing Guide](CONTRIBUTING.md) for how to run our formatting scripts.

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

❌ `tf.keras.activations.X`
✅ `keras.activations.X`

❌ `layers.X`
✅ `keras.layers.X` or `keras_nlp.layers.X`

❌ `Dense(1, activation='softmax')`
✅ `keras.layers.Dense(1, activation='softmax')`

For KerasNLP library code, `keras_nlp` will not be directly imported, but
`keras` should still be as a top-level object used to access library symbols.

## Ideal layer style

When writing a new KerasNLP layer (or tokenizer), please make sure to do the
following:

- Accept `**kwargs` in `__init__` and forward this to the super class.
- Keep a python attribute on the layer for each `__init__` argument to the
  layer. The name and value should match the passed value.
- Write a `get_config()` which chains to super.
- Document the layer behavior thouroughly including call behavior, on the
  class
- Always include usage examples including the full symbol location.

````python
class Linear(keras.layers.Layer):
    """A simple WX + B linear layer.

    This layer contains two trainable parameters, a weight matrix and bias
    vector. The layer will linearly transform input to an output of `units`
    size.

    Args:
        units: The dimensionality of the output space.

    Examples:

    Build a linear model.
    ```python
    inputs = keras.Input(shape=(2,))
    outputs = keras_nlp.layers.Linear(4)(inputs)
    model = keras.Model(inputs, outputs)
    ```

    Call the layer on direct input.
    >>> layer = keras_nlp.layers.Linear(4)
    >>> layer(tf.zeros(8, 2)) == layer.b
    True
    """
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        super().build(input_shape)
        self.w = self.add_weight(shape=(input_shape[-1], self.units))
        self.b = self.add_weight(shape=(self.units,))

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config
````
