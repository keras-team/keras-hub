from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.models.mpnet.mpnet_tokenizer import MPNetTokenizer
from keras_hub.src.models.text_embedder_preprocessor import (
    TextEmbedderPreprocessor,
)


@keras_hub_export("keras_hub.models.MPNetTextEmbedderPreprocessor")
class MPNetTextEmbedderPreprocessor(TextEmbedderPreprocessor):
    """An MPNet preprocessing layer which tokenizes and packs inputs for
    sentence embedding.

    This preprocessing layer will do three things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.MultiSegmentPacker`
       with the appropriate `"<s>"`, `"</s>"` and `"<pad>"` tokens, i.e.,
       adding a single `"<s>"` at the start of the entire sequence,
       `"</s></s>"` between segments, and a `"</s>"` at the end of the
       entire sequence.
    3. Construct a dictionary with keys `"token_ids"` and `"padding_mask"`
       that can be passed directly to an MPNet model.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    Args:
        tokenizer: A `keras_hub.models.MPNetTokenizer` instance.
        sequence_length: The length of the packed inputs. Defaults to 256.
            When loading a HuggingFace sentence-transformer preset, this is
            read from `sentence_bert_config.json` (e.g. 384 for
            `all-mpnet-base-v2`).
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
    preprocessor = keras_hub.models.MPNetTextEmbedderPreprocessor.from_preset(
        "all_mpnet_base_v2_en"
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of single sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.MPNetTextEmbedderPreprocessor.from_preset(
        "all_mpnet_base_v2_en"
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])

    # Map unlabeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = MPNetBackbone
    tokenizer_cls = MPNetTokenizer

    def build(self, input_shape):
        # MPNet doubles up the separator token between segments, like RoBERTa,
        # so we override build.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            sep_value=[self.tokenizer.end_token_id] * 2,
            pad_value=self.tokenizer.pad_token_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )
        self.built = True

    def _format_output(self, token_ids, segment_ids):
        """Add the padding mask. MPNet does not use segment ids."""
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
