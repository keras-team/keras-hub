from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.models.text_embedder_preprocessor import (
    TextEmbedderPreprocessor,
)


@keras_hub_export("keras_hub.models.Gemma3TextEmbedderPreprocessor")
class Gemma3TextEmbedderPreprocessor(TextEmbedderPreprocessor):
    """A Gemma3 preprocessing layer which tokenizes and packs inputs for
    sentence embedding.

    This preprocessing layer will do three things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.MultiSegmentPacker`
       with the appropriate `<bos>`, `<eos>`, and `<pad>` tokens.
    3. Construct a dictionary with keys `"token_ids"` and `"padding_mask"`
       that can be passed directly to a `Gemma3TextEmbedder` model.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    Args:
        tokenizer: A `keras_hub.models.Gemma3Tokenizer` instance.
        sequence_length: int. The length of the packed inputs. Defaults to
            `256`.
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
            unbatched. For single sequences, raw python inputs will be
            converted to tensors. For multiple sequences, pass tensors
            directly.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through
            unaltered.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_hub.models.Gemma3TextEmbedderPreprocessor.from_preset(
        "harrier_embedding_oss_270m",
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of single sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.Gemma3TextEmbedderPreprocessor.from_preset(
        "harrier_embedding_oss_270m",
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])

    # Map unlabeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = Gemma3Backbone
    tokenizer_cls = Gemma3Tokenizer

    def _format_output(self, token_ids, segment_ids):
        """Format packer output into Gemma3-compatible input dictionary."""
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
