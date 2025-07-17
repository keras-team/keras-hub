import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.esm.esm_backbone import ESMBackbone
from keras_hub.src.models.esm.esm_tokenizer import ESMTokenizer
from keras_hub.src.models.text_classifier_preprocessor import (
    TextClassifierPreprocessor,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.ESMProteinClassifierPreprocessor")
class ESMProteinClassifierPreprocessor(TextClassifierPreprocessor):
    """A ESM preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.StartEndPacker`.
       with the appropriate start, end and pad tokens.
    3. Construct a dictionary with the key `"token_ids"`, that can be passed
       directly to an ESM model.
    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    Args:
        tokenizer: A `keras_hub.models.ESMTokenizer` instance.
        sequence_length: The length of the packed inputs.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
            - `"round_robin"`: Available space is assigned one token at a
                time in a round-robin fashion to the inputs that still need
                some, until the limit is reached.
            - `"waterfall"`: The allocation of the budget is done using a
                "waterfall" algorithm that allocates quota in a
                left-to-right manner and fills up the buckets until we run
                out of budget. It supports an arbitrary number of segments.

    Call arguments:
        x: A tensor of single string sequences, or a tuple of multiple
            tensor sequences to be packed together. Inputs may be batched or
            unbatched. For single sequences, raw python inputs will be converted
            to tensors. For multiple sequences, pass tensors directly.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through unaltered.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_hub.models.ESMProteinClassifierPreprocessor.from_preset
    (
        hf://facebook/esm2_t6_8M_UR50D
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of single sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Preprocess a batch of sentence pairs.
    # When handling multiple sequences, always convert to tensors first!
    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    second = tf.constant(["The fox tripped.", "Oh look, a whale."])
    preprocessor((first, second))

    # Custom vocabulary.
    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]
    tokenizer = keras_hub.models.ESMTokenizer(vocabulary=vocab)
    preprocessor =
        keras_hub.models.ESMProteinClassifierPreprocessor(tokenizer)
    preprocessor("The quick brown fox jumped.")
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.ESMProteinClassifierPreprocessor.from_preset
    (
        hf://facebook/esm2_t6_8M_UR50D
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    second = tf.constant(["The fox tripped.", "Oh look, a whale."])
    label = tf.constant([1, 1])

    # Map labeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices((first, label))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map labeled sentence pairs.
    ds = tf.data.Dataset.from_tensor_slices(((first, second), label))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled sentence pairs.
    ds = tf.data.Dataset.from_tensor_slices((first, second))
    # Watch out for tf.data's default unpacking of tuples here!
    # Best to invoke the `preprocessor` directly in this case.
    ds = ds.map(
        lambda first, second: preprocessor(x=(first, second)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ```
    """

    backbone_cls = ESMBackbone
    tokenizer_cls = ESMTokenizer

    def build(self, input_shape):
        super().build(input_shape)
        # Defer masker creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
        )

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        x = self.tokenizer(x)
        token_ids = self.packer(x)
        x = {
            "token_ids": token_ids,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
