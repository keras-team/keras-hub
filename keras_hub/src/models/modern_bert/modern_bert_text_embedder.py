from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modern_bert.modern_bert_text_embedder_preprocessor import (  # noqa: E501
    ModernBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.text_embedder import TextEmbedder


@keras_hub_export("keras_hub.models.ModernBertTextEmbedder")
class ModernBertTextEmbedder(TextEmbedder):
    """An end-to-end ModernBERT model for generating sentence embeddings.

    This model attaches a pooling and optional L2 normalization head to a
    `keras_hub.models.ModernBertBackbone` instance, mapping from the backbone
    outputs to fixed-size sentence embeddings suitable for semantic
    similarity, clustering, and retrieval tasks.

    This is the architecture used by ModernBERT-based sentence-transformer
    models such as `nomic-ai/modernbert-embed-base` (mean pooling with L2
    normalization) and `Alibaba-NLP/gte-modernbert-base` (`[CLS]` pooling
    without normalization). When loading a Hugging Face sentence-transformers
    checkpoint with `from_preset()`, the pooling strategy, normalization, and
    sequence length are read from the checkpoint's sentence-transformers
    config files.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Task prefixes (prompts) are not added automatically. Some checkpoints
    expect them on their inputs, e.g. `nomic-ai/modernbert-embed-base`
    expects `"search_query: "` on queries and `"search_document: "` on
    documents. Prepend them to the input strings yourself, as the examples
    below do. Without them, the embeddings are silently worse.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.ModernBertBackbone` instance.
        preprocessor: A `keras_hub.models.ModernBertTextEmbedderPreprocessor`
            or `None`. If `None`, this model will not apply preprocessing,
            and inputs should be preprocessed before calling the model.
        pooling_mode: str. The pooling strategy to use. One of `"mean"`,
            `"cls"`, or `"max"`. Defaults to `"mean"`.
            - `"mean"`: Attention-mask-aware mean pooling over all tokens.
            - `"cls"`: Use the `[CLS]` token representation.
            - `"max"`: Max pooling over all tokens, ignoring padding.
        normalize: bool. Whether to L2 normalize the output embeddings.
            Defaults to `True`.

    Examples:

    Raw string data.
    ```python
    # Inputs are padded to the checkpoint's 8192-token limit by default.
    # Load the preprocessor with a shorter length for faster inference on
    # short texts. On an already loaded model, setting
    # `embedder.preprocessor.sequence_length = 512` works too.
    preprocessor = (
        keras_hub.models.ModernBertTextEmbedderPreprocessor.from_preset(
            "hf://nomic-ai/modernbert-embed-base",
            sequence_length=512,
        )
    )
    embedder = keras_hub.models.ModernBertTextEmbedder.from_preset(
        "hf://nomic-ai/modernbert-embed-base",
        preprocessor=preprocessor,
    )

    # Semantic search. This checkpoint expects task prefixes, which are not
    # added for you.
    query = "search_query: Which planet is known as the Red Planet?"
    documents = [
        "search_document: Mars is often referred to as the Red Planet.",
        "search_document: Venus is often called Earth's twin.",
    ]
    q_emb = embedder.encode_text(query)
    d_embs = embedder.encode_documents(documents)
    sims = embedder.similarity(q_emb, d_embs)
    print("Best match:", documents[ops.convert_to_numpy(sims).argmax()])
    ```

    Preprocessed integer data.
    ```python
    features = {
        "token_ids": np.ones(shape=(2, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
    }

    embedder = keras_hub.models.ModernBertTextEmbedder.from_preset(
        "hf://nomic-ai/modernbert-embed-base",
        preprocessor=None,
    )
    embeddings = embedder.predict(features)
    ```
    """

    backbone_cls = ModernBertBackbone
    preprocessor_cls = ModernBertTextEmbedderPreprocessor

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
        # `ModernBertBackbone` returns the final hidden states directly,
        # rather than a dictionary of outputs.
        sequence_output = backbone(inputs)
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

        Both sums are accumulated in `float32` and the mean is cast back to
        the input dtype. In `float16`, summing thousands of tokens overflows
        to `inf` or stalls, and the presets accept up to 8192 tokens.
        """
        dtype = sequence_output.dtype
        mask = ops.cast(ops.expand_dims(padding_mask, axis=-1), "float32")
        sequence_output = ops.cast(sequence_output, "float32")
        sum_embeddings = ops.sum(sequence_output * mask, axis=1)
        sum_mask = ops.maximum(ops.sum(mask, axis=1), 1e-5)
        return ops.cast(sum_embeddings / sum_mask, dtype)

    @staticmethod
    def _max_pooling(sequence_output, padding_mask):
        """Max pooling over token embeddings, ignoring padding."""
        mask = ops.cast(ops.expand_dims(padding_mask, axis=-1), dtype="bool")
        fill_value = ops.cast(
            ops.convert_to_tensor(float("-inf")), sequence_output.dtype
        )
        masked_output = ops.where(mask, sequence_output, fill_value)
        return ops.max(masked_output, axis=1)

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
            **kwargs: Additional keyword arguments passed to `predict()`.

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
