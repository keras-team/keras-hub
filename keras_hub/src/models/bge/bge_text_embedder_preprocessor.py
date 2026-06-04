from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.bge.bge_backbone import BgeBackbone
from keras_hub.src.models.bge.bge_tokenizer import BgeTokenizer
from keras_hub.src.models.text_classifier_preprocessor import (
    TextClassifierPreprocessor,
)


@keras_hub_export(
    [
        "keras_hub.models.BgeTextEmbedderPreprocessor",
        "keras_hub.models.BgeEmbedderPreprocessor",
    ]
)
class BgeTextEmbedderPreprocessor(TextClassifierPreprocessor):
    """A BGE preprocessing layer which tokenizes and packs inputs for embedding.

    This preprocessing layer will do three things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.MultiSegmentPacker`
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
    3. Construct a dictionary with keys `"token_ids"`, `"segment_ids"`, and
       `"padding_mask"` that can be passed directly to a `BgeBackbone`.

    This layer is intended to be paired with `BgeTextEmbedder`. The
    preprocessing is identical to standard BERT classifier preprocessing but is
    provided as a separate class so that `from_preset()` on
    `BgeTextEmbedder` automatically resolves the correct preprocessor.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    Args:
        tokenizer: A `keras_hub.models.BgeTokenizer` instance.
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
    preprocessor = keras_hub.models.BgeTextEmbedderPreprocessor.from_preset(
        "bge_small_en_v1.5"
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and pack a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Tokenize and pack a query/passage pair.
    preprocessor(("query text", "passage text"))
    ```
    """

    backbone_cls = BgeBackbone
    tokenizer_cls = BgeTokenizer
