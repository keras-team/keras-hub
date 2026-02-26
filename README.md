---
library_name: keras-hub
license: apache-2.0
tags:
- text-classification
- keras
pipeline_tag: text-classification
---
### Model Overview
An XLM-RoBERTa encoder network.

This class implements a bi-directional Transformer-based encoder as
described in ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
It includes the embedding lookups and transformer layers, but it does not
include the masked language modeling head used during pretraining.

The default constructor gives a fully customizable, randomly initialized
RoBERTa encoder with any number of layers, heads, and embedding dimensions.
To load preset architectures and weights, use the `from_preset()`
constructor.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind. The underlying model is provided by a
third party and subject to a separate license, available
[here](https://github.com/facebookresearch/fairseq).

## Links

* [XLM-RoBERTa Quickstart Notebook](https://www.kaggle.com/code/laxmareddypatlolla/xlm-roberta-quickstart-notebook)
* [XLM-RoBERTa  API Documentation](https://keras.io/keras_hub/api/models/xlm_roberta/)
* [XLM-RoBERTa  Model Card](https://huggingface.co/FacebookAI/xlm-roberta-base)
* [KerasHub Beginner Guide](https://keras.io/guides/keras_hub/getting_started/)
* [KerasHub Model Publishing Guide](https://keras.io/guides/keras_hub/upload/)

## Installation

Keras and KerasHub can be installed with:

```
pip install -U -q keras-hub
pip install -U -q keras
```

Jax, TensorFlow, and Torch come preinstalled in Kaggle Notebooks. For instructions on installing them in another environment see the [Keras Getting Started](https://keras.io/getting_started/) page.

## Presets

The following model checkpoints are provided by the Keras team. Full code examples for each are available below.
| Preset name    | Parameters | Description                                      |
|----------------|------------|--------------------------------------------------|
| xlm_roberta_base_multi |   277.45M  | 12-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages.|
| xlm_roberta_large_multi | 558.84M  | 24-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages. |


__Arguments__


- __vocabulary_size__: int. The size of the token vocabulary.
- __num_layers__: int. The number of transformer layers.
- __num_heads__: int. The number of attention heads for each transformer.
    The hidden size must be divisible by the number of attention heads.
- __hidden_dim__: int. The size of the transformer encoding layer.
- __intermediate_dim__: int. The output dimension of the first Dense layer in
    a two-layer feedforward network for each transformer.
- __dropout__: float. Dropout probability for the Transformer encoder.
- __max_sequence_length__: int. The maximum sequence length this encoder can
    consume. The sequence length of the input must be less than
    `max_sequence_length` default value. This determines the variable
    shape for positional embeddings.

## Example Usage
```python
import keras
import keras_hub
import numpy as np
```

Raw string data.
```python
features = ["The quick brown fox jumped.", "نسيت الواجب"]
labels = [0, 3]

# Pretrained classifier.
classifier = keras_hub.models.XLMRobertaClassifier.from_preset(
    "xlm_roberta_large_multi",
    num_classes=4,
)
classifier.fit(x=features, y=labels, batch_size=2)
classifier.predict(x=features, batch_size=2)

# Re-compile (e.g., with a new learning rate).
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
)
# Access backbone programmatically (e.g., to change `trainable`).
classifier.backbone.trainable = False
# Fit again.
classifier.fit(x=features, y=labels, batch_size=2)
```

Preprocessed integer data.
```python
features = {
    "token_ids": np.ones(shape=(2, 12), dtype="int32"),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
}
labels = [0, 3]

# Pretrained classifier without preprocessing.
classifier = keras_hub.models.XLMRobertaClassifier.from_preset(
    "xlm_roberta_large_multi",
    num_classes=4,
    preprocessor=None,
)
classifier.fit(x=features, y=labels, batch_size=2)
```

## Example Usage with Hugging Face URI

```python
import keras
import keras_hub
import numpy as np
```

Raw string data.
```python
features = ["The quick brown fox jumped.", "نسيت الواجب"]
labels = [0, 3]

# Pretrained classifier.
classifier = keras_hub.models.XLMRobertaClassifier.from_preset(
    "hf://keras/xlm_roberta_large_multi",
    num_classes=4,
)
classifier.fit(x=features, y=labels, batch_size=2)
classifier.predict(x=features, batch_size=2)

# Re-compile (e.g., with a new learning rate).
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
)
# Access backbone programmatically (e.g., to change `trainable`).
classifier.backbone.trainable = False
# Fit again.
classifier.fit(x=features, y=labels, batch_size=2)
```

Preprocessed integer data.
```python
features = {
    "token_ids": np.ones(shape=(2, 12), dtype="int32"),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
}
labels = [0, 3]

# Pretrained classifier without preprocessing.
classifier = keras_hub.models.XLMRobertaClassifier.from_preset(
    "hf://keras/xlm_roberta_large_multi",
    num_classes=4,
    preprocessor=None,
)
classifier.fit(x=features, y=labels, batch_size=2)
```
