import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
    LayoutLMv3Backbone,
)
from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import (
    LayoutLMv3Tokenizer,
)
from keras_hub.src.models.preprocessor import Preprocessor


@keras_hub_export("keras_hub.models.LayoutLMv3DocumentClassifierPreprocessor")
class LayoutLMv3DocumentClassifierPreprocessor(Preprocessor):
    """LayoutLMv3 preprocessor for document classification tasks.

    This preprocessing layer is meant for use with
    `keras_hub.models.LayoutLMv3Backbone`, and can be used to chain a
    `keras_hub.models.LayoutLMv3Tokenizer` with the model preprocessing logic.
    It can optionally be configured with a `sequence_length` which will pad or
    truncate sequences to a fixed length.

    Arguments:
        tokenizer: A `keras_hub.models.LayoutLMv3Tokenizer` instance.
        sequence_length: int. If set, the output will be packed or padded to
            exactly this sequence length.

    Call arguments:
        x: A dictionary with "text" and optionally "bbox" keys. The "text"
            should be a string or tensor of strings. The "bbox" should be a
            list or tensor of bounding box coordinates with shape
            `(..., num_words, 4)`.
        y: Label data. Should always be `None` as the layer is unsupervised.
        sample_weight: Label weights. Should always be `None` as the layer is
            unsupervised.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = (
        keras_hub.models.LayoutLMv3DocumentClassifierPreprocessor.from_preset(
            "layoutlmv3_base"
        )
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Tokenize with bounding boxes.
    preprocessor({
        "text": "Hello world",
        "bbox": [[0, 0, 100, 50], [100, 0, 200, 50]]
    })
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = (
        keras_hub.models.LayoutLMv3DocumentClassifierPreprocessor.from_preset(
            "layoutlmv3_base"
        )
    )

    text_ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox."])
    text_ds = text_ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = LayoutLMv3Backbone
    tokenizer_cls = LayoutLMv3Tokenizer

    def call(self, x, y=None, sample_weight=None):
        if isinstance(x, dict):
            text = x["text"]
            bbox = x.get("bbox", None)
        else:
            text = x
            bbox = None

        token_output = self.tokenizer(
            text, bbox=bbox, sequence_length=self.sequence_length
        )

        # The tokenizer already provides token_ids, padding_mask, and bbox
        # Rename token_ids to match backbone expectations
        output = {
            "token_ids": token_output["token_ids"],
            "padding_mask": token_output["padding_mask"],
            "bbox": token_output["bbox"],
        }

        return keras.utils.pack_x_y_sample_weight(output, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        return config
