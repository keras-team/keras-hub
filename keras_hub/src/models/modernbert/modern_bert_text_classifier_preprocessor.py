import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.models.text_classifier_preprocessor import (
    TextClassifierPreprocessor,
)


@keras_hub_export(
    [
        "keras_hub.models.ModernBertTextClassifierPreprocessor",
        "keras_hub.models.ModernBertPreprocessor",
    ]
)
class ModernBertTextClassifierPreprocessor(TextClassifierPreprocessor):
    """A ModernBERT preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer:

    1. Tokenizes the input text using `ModernBertTokenizer`.
    2. Adds the ModernBERT start and end tokens.
    3. Pads or truncates the resulting sequence to `sequence_length`.
    4. Returns `token_ids` and `padding_mask`, which can be passed directly
       to `ModernBertBackbone`.

    ModernBERT does not use segment IDs, so they are removed from the
    output dictionary.

    Args:
        tokenizer: A `keras_hub.models.ModernBertTokenizer` instance.
        sequence_length: int. The length of the packed sequence.
        truncate: string. The truncation algorithm to use. Supported values
            are `"round_robin"` and `"waterfall"`.
        **kwargs: Additional keyword arguments passed to
            `TextClassifierPreprocessor`.

    Examples:
    ```python
    tokenizer = keras_hub.models.ModernBertTokenizer(
        vocabulary=vocabulary,
        merges=merges,
    )

    preprocessor = (
        keras_hub.models.ModernBertTextClassifierPreprocessor(
            tokenizer=tokenizer,
            sequence_length=128,
        )
    )

    x = preprocessor("The quick brown fox.")
    ```
    """

    backbone_cls = ModernBertBackbone
    tokenizer_cls = ModernBertTokenizer

    def build(self, input_shape):
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            sep_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )
        self.built = True

    def call(self, x, y=None, sample_weight=None):
        output = super().call(
            x,
            y=y,
            sample_weight=sample_weight,
        )

        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)

        # ModernBERT does not use segment IDs.
        if "segment_ids" in x:
            del x["segment_ids"]

        return keras.utils.pack_x_y_sample_weight(
            x,
            y,
            sample_weight,
        )
