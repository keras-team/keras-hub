from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_text_embedder_preprocessor import (
    BertTextEmbedderPreprocessor,
)
from keras_hub.src.models.text_embedder import TextEmbedder


@keras_hub_export("keras_hub.models.BertTextEmbedder")
class BertTextEmbedder(TextEmbedder):
    """An end-to-end BERT model for generating sentence embeddings.

    This model attaches a mean pooling and L2 normalization head to a
    `keras_hub.models.BertBackbone` instance, mapping from the backbone
    outputs to fixed-size sentence embeddings suitable for semantic
    similarity, clustering, and retrieval tasks.

    This is the architecture used by sentence-transformers models like
    `all-MiniLM-L6-v2`. For usage of this model with pre-trained weights,
    use the `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.BertBackbone` instance.
        preprocessor: A `keras_hub.models.BertTextEmbedderPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        pooling_mode: str. The pooling strategy to use. One of `"mean"`,
            `"cls"`, or `"max"`. Defaults to `"mean"`.
            - `"mean"`: Attention-mask-aware mean pooling over all tokens.
            - `"cls"`: Use the `[CLS]` token representation.
            - `"max"`: Max pooling over all tokens.
        normalize: bool. Whether to L2 normalize the output embeddings.
            Defaults to `True`.

    Examples:

    Raw string data.
    ```python
    embedder = keras_hub.models.BertTextEmbedder.from_preset(
        "all_minilm_l6_v2_en",
    )
    embeddings = embedder.predict(
        ["The quick brown fox.", "I forgot my homework."]
    )
    # embeddings.shape == (2, 384)

    # Compute cosine similarity.
    similarity = embeddings @ embeddings.T
    ```

    Preprocessed integer data.
    ```python
    features = {
        "token_ids": np.ones(shape=(2, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
    }

    embedder = keras_hub.models.BertTextEmbedder.from_preset(
        "all_minilm_l6_v2_en",
        preprocessor=None,
    )
    embeddings = embedder.predict(features)
    ```
    """

    backbone_cls = BertBackbone
    preprocessor_cls = BertTextEmbedderPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        pooling_mode="mean",
        normalize=True,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        backbone_outputs = backbone(inputs)
        sequence_output = backbone_outputs["sequence_output"]
        padding_mask = inputs["padding_mask"]

        # Apply pooling.
        if pooling_mode == "mean":
            pooled = self._mean_pooling(sequence_output, padding_mask)
        elif pooling_mode == "cls":
            pooled = sequence_output[:, 0, :]
        elif pooling_mode == "max":
            pooled = self._max_pooling(sequence_output, padding_mask)
        else:
            raise ValueError(
                f"Invalid pooling_mode: '{pooling_mode}'. "
                "Expected one of 'mean', 'cls', or 'max'."
            )

        # Apply L2 normalization.
        if normalize:
            pooled = self._l2_normalize(pooled)

        super().__init__(
            inputs=inputs,
            outputs=pooled,
            **kwargs,
        )

        # === Config ===
        self.pooling_mode = pooling_mode
        self.normalize = normalize

    @staticmethod
    def _mean_pooling(sequence_output, padding_mask):
        """Attention-mask-aware mean pooling over token embeddings."""
        # Expand mask: [batch, seq_len] -> [batch, seq_len, 1]
        mask = ops.cast(
            ops.expand_dims(padding_mask, axis=-1), sequence_output.dtype
        )
        # Sum token embeddings, masked.
        sum_embeddings = ops.sum(sequence_output * mask, axis=1)
        # Sum mask for normalization.
        sum_mask = ops.maximum(ops.sum(mask, axis=1), 1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def _max_pooling(sequence_output, padding_mask):
        """Max pooling over token embeddings, ignoring padding."""
        mask = ops.cast(
            ops.expand_dims(padding_mask, axis=-1), sequence_output.dtype
        )
        # Set padding positions to -inf so they don't affect max.
        masked_output = sequence_output + (1.0 - mask) * (-1e9)
        return ops.max(masked_output, axis=1)

    @staticmethod
    def _l2_normalize(embeddings):
        """L2 normalize embeddings to unit length."""
        norm = ops.sqrt(
            ops.maximum(
                ops.sum(ops.square(embeddings), axis=-1, keepdims=True), 1e-12
            )
        )
        return embeddings / norm

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pooling_mode": self.pooling_mode,
                "normalize": self.normalize,
            }
        )
        return config
