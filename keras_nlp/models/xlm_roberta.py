# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XLM-RoBERTa model configurable class, preconfigured versions, and task heads."""

from tensorflow import keras

from keras_nlp.models import roberta
from keras_nlp.models.roberta import RobertaMultiSegmentPacker
from keras_nlp.tokenizers import SentencePieceTokenizer


class XLMRobertaCustom(roberta.RobertaCustom):
    """XLM-RoBERTa encoder with a customizable set of hyperparameters.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    This class gives a fully configurable XLM-R model with any number of
    layers, heads, and embedding dimensions. For specific XLM-R architectures
    defined in the paper, see, for example, `keras_nlp.models.XLMRoberta`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length`.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized XLM-R model
    model = keras_nlp.models.XLMRobertaCustom(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call encoder on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=50265),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    output = model(input_data)
    ```
    """

    pass


PREPROCESSOR_DOCSTRING = """XLM-RoBERTa preprocessor with pretrained vocabularies.

This preprocessing layer will do three things:

 - Tokenize any number of inputs using a
   `keras_nlp.tokenizers.SentencePieceTokenizer`.
 - Pack the inputs together using a
   `keras_nlp.models.roberta.RobertaMultiSegmentPacker` with the appropriate
   `"<s>"`, `"</s>"` and `"<pad>"` tokens, i.e., adding a single `<s>` at the
   start of the entire sequence, `[</s>, </s>]` at the end of each segment,
   save the last and a `</s>` at the end of the entire sequence.
 - Construct a dictionary of with keys `"token_ids"`, `"padding_mask"`, that can
   be passed directly to a XLM-RoBERTa model.

This layer will accept either a tuple of (possibly batched) inputs, or a single
input tensor. If a single tensor is passed, it will be packed equivalently to
a tuple with a single element.

The SentencePiece tokenizer can be accessed via the `tokenizer` property on this
layer, and can be used directly for custom packing on inputs.

Args:
    spm_proto: Either a `string` path to a SentencePiece proto file, or a
               `bytes` object with a serialized SentencePiece proto. See the
               [SentencePiece repository](https://github.com/google/sentencepiece)
               for more details on the format.
    sequence_length: The length of the packed inputs. Only used if
        `pack_inputs` is True.
    truncate: The algorithm to truncate a list of batched segments to fit
        within `sequence_length`. Only used if
        `pack_inputs` is True. The value can be either `round_robin` or
        `waterfall`:
            - `"round_robin"`: Available space is assigned one token at a
                time in a round-robin fashion to the inputs that still need
                some, until the limit is reached.
            - `"waterfall"`: The allocation of the budget is done using a
                "waterfall" algorithm that allocates quota in a
                left-to-right manner and fills up the buckets until we run
                out of budget. It supports an arbitrary number of segments.

Examples:
```python
preprocessor = keras_nlp.models.XLMRobertaPreprocessor(spm_proto="model.spm")

# Tokenize and pack a single sentence directly.
preprocessor("The quick brown fox jumped.")

# Tokenize and pack a multiple sentence directly.
preprocessor(("The quick brown fox jumped.", "Call me Ishmael."))

# Map a dataset to preprocess a single sentence.
features = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 1]
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.map(
    lambda x, y: (preprocessor(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Map a dataset to preprocess a multiple sentences.
first_sentences = ["The quick brown fox jumped.", "Call me Ishmael."]
second_sentences = ["The fox tripped.", "Oh look, a whale."]
labels = [1, 1]
ds = tf.data.Dataset.from_tensor_slices(((first_sentences, second_sentences), labels))
ds = ds.map(
    lambda x, y: (preprocessor(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
```
"""


class XLMRobertaPreprocessor(keras.layers.Layer):
    def __init__(
        self,
        spm_proto,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # TODO(abheesht17): Instead of passing `spm_proto`, do something similar
        # to `BertPreprocessor` once weights have been uploaded.

        self.tokenizer = SentencePieceTokenizer(proto=spm_proto)

        # Check for necessary special tokens.
        start_token = "<s>"
        end_token = "</s>"
        pad_token = "<pad>"
        for token in [start_token, end_token, pad_token]:
            if token not in self.tokenizer.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.packer = RobertaMultiSegmentPacker(
            start_value=self.tokenizer.token_to_id(start_token),
            end_value=self.tokenizer.token_to_id(end_token),
            pad_value=self.tokenizer.token_to_id(pad_token),
            truncate=truncate,
            sequence_length=sequence_length,
        )

    def vocabulary_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.vocabulary_size()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spm_proto": self.tokenizer.proto,
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [self.tokenizer(x) for x in inputs]
        token_ids, segment_ids = self.packer(inputs)
        return {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "padding_mask": token_ids != 0,
        }


# TODO(abheesht17): Will have to add args to `PREPROCESSOR_DOCSTRING` once
# weights are uploaded.
setattr(
    XLMRobertaPreprocessor,
    "__doc__",
    PREPROCESSOR_DOCSTRING,
)


MODEL_DOCSTRING = """XLM-RoBERTa "{type}" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    Args:
        vocabulary_size: int, optional. The size of the token vocabulary.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized XLMRoberta{type} encoder
    model = keras_nlp.models.XLMRoberta{type}(vocabulary_size=10000)

    # Call encoder on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.ones((1, 512)),
    }
    output = model(input_data)
    ```
"""


def XLMRobertaBase(vocabulary_size, name=None, trainable=True):
    return XLMRobertaCustom(
        vocabulary_size=vocabulary_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )


def XLMRobertaLarge(vocabulary_size, name=None, trainable=True):
    return XLMRobertaCustom(
        vocabulary_size=vocabulary_size,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        intermediate_dim=4096,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )
