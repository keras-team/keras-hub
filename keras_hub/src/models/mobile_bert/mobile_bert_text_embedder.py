from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_embedder_preprocessor import (  # noqa: E501
    MobileBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.text_embedder import TextEmbedder


@keras_hub_export("keras_hub.models.MobileBertTextEmbedder")
class MobileBertTextEmbedder(TextEmbedder):
    """An end-to-end MobileBERT model for generating sentence embeddings.

    This model attaches a mean pooling and L2 normalization head to a
    `keras_hub.models.MobileBertBackbone` instance, mapping from the backbone
    outputs to fixed-size sentence embeddings suitable for semantic
    similarity, clustering, and retrieval tasks.

    Note: The base `mobilebert_uncased` checkpoint was trained for masked
    language modeling and next sentence prediction. Its raw pooled embeddings
    will collapse to a single direction (cosine similarity ~1.0). This model
    should be fine-tuned on a contrastive loss before the embeddings are useful.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.MobileBertBackbone` instance.
        preprocessor: A `keras_hub.models.MobileBertTextEmbedderPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model. Defaults to
            `None`.
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
    embedder = keras_hub.models.MobileBertTextEmbedder.from_preset(
        "mobilebert_uncased",
    )

    # Semantic search.
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Mars is often referred to as the Red Planet.",
        "Venus is often called Earth's twin.",
    ]
    q_emb = embedder.encode_text(query)
    d_embs = embedder.encode_documents(documents)
    sims = embedder.similarity(q_emb, d_embs)
    print("Best match:", documents[keras.ops.convert_to_numpy(sims).argmax()])
    ```

    Preprocessed integer data.
    ```python
    features = {
        "token_ids": np.ones(shape=(2, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
    }

    embedder = keras_hub.models.MobileBertTextEmbedder.from_preset(
        "mobilebert_uncased",
        preprocessor=None,
    )
    embeddings = embedder.predict(features)
    ```
    """

    backbone_cls = MobileBertBackbone
    preprocessor_cls = MobileBertTextEmbedderPreprocessor

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
        """Attention-mask-aware mean pooling over token embeddings.

        Both sums are accumulated in `float32` and cast back to the input
        dtype: summing up to 512 tokens overflows `float16` (65504) once
        the mean token magnitude exceeds ~128.
        """
        dtype = sequence_output.dtype
        # Expand mask: [batch, seq_len] -> [batch, seq_len, 1]
        mask = ops.cast(ops.expand_dims(padding_mask, axis=-1), "float32")
        sequence_output = ops.cast(sequence_output, "float32")
        # Sum token embeddings, masked.
        sum_embeddings = ops.sum(sequence_output * mask, axis=1)
        # Sum mask for normalization.
        sum_mask = ops.maximum(ops.sum(mask, axis=1), 1.0)
        return ops.cast(sum_embeddings / sum_mask, dtype)

    @staticmethod
    def _max_pooling(sequence_output, padding_mask):
        """Max pooling over token embeddings, ignoring padding."""
        mask = ops.cast(ops.expand_dims(padding_mask, axis=-1), dtype="bool")
        # Set padding positions to -inf so they don't affect max.
        fill_value = ops.cast(
            ops.convert_to_tensor(float("-inf")), sequence_output.dtype
        )
        masked_output = ops.where(mask, sequence_output, fill_value)
        pooled = ops.max(masked_output, axis=1)
        # Rows with no valid token would otherwise be `-inf` (and `NaN`
        # after normalization); return zeros, matching `_mean_pooling`.
        has_valid = ops.any(mask, axis=1)
        return ops.where(has_valid, pooled, ops.zeros_like(pooled))

    @staticmethod
    def _l2_normalize(embeddings):
        """L2 normalize embeddings to unit length.

        The norm is computed in `float32`, since the sum of squares can
        overflow in `float16`. The result is cast back to the input dtype:
        `ops.nn.normalize` reports a `float32` symbolic output for any input
        dtype, which would otherwise make the functional model's output spec
        disagree with its eager output under mixed precision.
        """
        dtype = embeddings.dtype
        embeddings = ops.cast(embeddings, "float32")
        normalized = ops.nn.normalize(embeddings, axis=-1, order=2)
        return ops.cast(normalized, dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pooling_mode": self.pooling_mode,
                "normalize": self.normalize,
            }
        )
        return config

    def encode_documents(self, documents, **kwargs):
        """Encode a string or list of documents into embeddings.

        This is a convenience method that wraps `predict()` for a
        single document or batch of documents. The output embeddings
        are suitable for computing similarity against query embeddings.

        Args:
            documents: A string or list of strings to encode.
            **kwargs: Additional keyword arguments passed to
                `predict()`.

        Returns:
            A tensor of shape `(batch_size, embedding_dim)`.
        """
        if isinstance(documents, str):
            documents = [documents]
        return self.predict(documents, **kwargs)

    def similarity(self, query_embeddings, document_embeddings):
        """Compute similarity between query and document embeddings.

        Computes the dot product between query and document embeddings.
        When embeddings are L2 normalized (the default), this is
        equivalent to cosine similarity. Returns a similarity matrix
        of shape `(num_queries, num_documents)`.

        Args:
            query_embeddings: An array or tensor of shape
                `(num_queries, embedding_dim)`.
            document_embeddings: An array or tensor of shape
                `(num_documents, embedding_dim)`.

        Returns:
            A tensor of shape `(num_queries, num_documents)` containing
            similarity scores.
        """
        query_tensor = ops.convert_to_tensor(query_embeddings)
        document_tensor = ops.convert_to_tensor(document_embeddings)
        return ops.matmul(query_tensor, ops.transpose(document_tensor))
