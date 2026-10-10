from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder_preprocessor import (  # noqa: E501
    EmbeddingGemma2TextEmbedderPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4MeanPooling
from keras_hub.src.models.text_embedder import TextEmbedder


@keras_hub_export("keras_hub.models.EmbeddingGemma2TextEmbedder")
class EmbeddingGemma2TextEmbedder(TextEmbedder):
    """An end-to-end EmbeddingGemma2 model for generating sentence embeddings.

    This model attaches a mean-pooling and L2 normalization head to an
    `EmbeddingGemma2Backbone` instance, mapping from the backbone outputs to
    fixed-size sentence embeddings suitable for semantic similarity, clustering,
    and retrieval tasks.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Args:
        backbone: An `keras_hub.models.EmbeddingGemma2Backbone` instance.
        preprocessor: An
            `keras_hub.models.EmbeddingGemma2TextEmbedderPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        pooling_mode: string. Kept for configuration compatibility with other
            embedders. Only `"mean"` is supported. Defaults to `"mean"`.
        normalize: bool. Whether to L2 normalize the output embeddings.
            Defaults to `True`.
    """

    backbone_cls = EmbeddingGemma2Backbone
    preprocessor_cls = EmbeddingGemma2TextEmbedderPreprocessor

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
        backbone_output = backbone(inputs)
        padding_mask = inputs["padding_mask"]

        if isinstance(backbone_output, dict):
            sequence_output = backbone_output["sequence_output"]
        else:
            sequence_output = backbone_output

        # Apply mean pooling
        if pooling_mode != "mean":
            raise ValueError(
                f"EmbeddingGemma2TextEmbedder only supports pooling_mode='mean'. "  # noqa: E501
                f"Received: pooling_mode={pooling_mode}"
            )

        pooled = Gemma4MeanPooling(dtype=backbone.dtype, name="mean_pooling")(
            sequence_output, padding_mask=padding_mask
        )

        # Apply L2 normalization
        if normalize:
            pooled = layers.UnitNormalization(
                axis=-1, dtype=backbone.dtype, name="unit_normalization"
            )(pooled)

        super().__init__(
            inputs=inputs,
            outputs=pooled,
            **kwargs,
        )

        # === Config ===
        self.pooling_mode = pooling_mode
        self.normalize = normalize

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pooling_mode": self.pooling_mode,
                "normalize": self.normalize,
            }
        )
        return config
