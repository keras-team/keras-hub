from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.models.text_embedder_preprocessor import (
    TextEmbedderPreprocessor,
)


@keras_hub_export("keras_hub.models.BertTextEmbedderPreprocessor")
class BertTextEmbedderPreprocessor(TextEmbedderPreprocessor):
    """A BERT preprocessing layer which tokenizes and packs inputs for
    sentence embedding.

    This preprocessing layer will do three things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.MultiSegmentPacker`
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
    3. Construct a dictionary with keys `"token_ids"`, `"segment_ids"`,
       `"padding_mask"`, that can be passed directly to a BERT model.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    Args:
        tokenizer: A `keras_hub.models.BertTokenizer` instance.
        sequence_length: The length of the packed inputs. Defaults to 256,
            which is the recommended max sequence length for sentence
            embedding models.
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
    preprocessor = keras_hub.models.BertTextEmbedderPreprocessor.from_preset(
        "all_minilm_l6_v2_en"
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of single sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.BertTextEmbedderPreprocessor.from_preset(
        "all_minilm_l6_v2_en"
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])

    # Map unlabeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = BertBackbone
    tokenizer_cls = BertTokenizer

    def _format_output(self, token_ids, segment_ids):
        """Add BERT-specific padding mask to preprocessor output."""
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
            "segment_ids": segment_ids,
        }
