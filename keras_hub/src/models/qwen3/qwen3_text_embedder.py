from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_text_embedder_preprocessor import (
    Qwen3TextEmbedderPreprocessor,
)
from keras_hub.src.models.text_embedder import TextEmbedder


@keras_hub_export("keras_hub.models.Qwen3TextEmbedder")
class Qwen3TextEmbedder(TextEmbedder):
    """An end-to-end Qwen3 model for generating sentence embeddings.

    This model attaches a pooling and optional L2 normalization head to a
    `keras_hub.models.Qwen3Backbone` instance, mapping from the backbone
    outputs to fixed-size sentence embeddings suitable for semantic
    similarity, clustering, and retrieval tasks.

    This is the architecture used by Microsoft's harrier-oss embedding models.
    For usage of this model with pre-trained weights, use the `from_preset()`
    constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.Qwen3Backbone` instance.
        preprocessor: A `keras_hub.models.Qwen3TextEmbedderPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        pooling_mode: string. The pooling strategy to use. One of `"last"` or
            `"mean"`. Defaults to `"last"`.
            - `"last"`: Use the hidden state of the last non-padding token.
              This is the standard pooling strategy for decoder-only models
              such as Qwen3 and matches the approach used by harrier-oss.
            - `"mean"`: Attention-mask-aware mean pooling over all tokens.
        normalize: bool. Whether to L2 normalize the output embeddings.
            Defaults to `True`.

    Examples:

    Raw string data.
    ```python
    embedder = keras_hub.models.Qwen3TextEmbedder.from_preset(
        "hf://microsoft/harrier-oss-v1-0.6b",
    )

    # Semantic search.
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Mars is often referred to as the Red Planet.",
        "Venus is often called Earth's twin.",
    ]
    q_emb = embedder.encode_text(
        "Instruct: Given a web search query, retrieve relevant passages
        \\nQuery: " + query
    )
    d_embs = embedder.encode_text(documents)
    sims = embedder.similarity(q_emb, d_embs)
    print("Best match:", documents[sims.numpy().argmax()])
    ```

    Preprocessed integer data.
    ```python
    features = {
        "token_ids": np.ones(shape=(2, 12), dtype="int32"),
        "padding_mask": np.ones(shape=(2, 12), dtype="int32"),
    }

    embedder = keras_hub.models.Qwen3TextEmbedder.from_preset(
        "hf://microsoft/harrier-oss-v1-0.6b",
        preprocessor=None,
    )
    embeddings = embedder.predict(features)
    ```

    Reference:
     - [harrier-oss-v1 model page](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
    """

    backbone_cls = Qwen3Backbone
    preprocessor_cls = Qwen3TextEmbedderPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        pooling_mode="last",
        normalize=True,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        # Qwen3Backbone outputs a raw sequence tensor of shape
        # (batch, seq_len, hidden_dim), not a dict.
        sequence_output = backbone(inputs)
        padding_mask = inputs["padding_mask"]

        # Apply pooling.
        if pooling_mode == "last":
            pooled = self._last_token_pooling(sequence_output, padding_mask)
        elif pooling_mode == "mean":
            pooled = self._mean_pooling(sequence_output, padding_mask)
        else:
            raise ValueError(
                f"Invalid pooling_mode: '{pooling_mode}'. "
                "Expected one of 'last' or 'mean'."
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
    def _last_token_pooling(sequence_output, padding_mask):
        """Pool the hidden state of the last non-padding token."""
        # padding_mask: (batch, seq_len), 1 for real tokens, 0 for padding.
        mask = ops.cast(padding_mask, sequence_output.dtype)
        # Pad one zero on the right, then slice off position 0 to shift left.
        # mask_shifted[i] = mask[i+1], with mask_shifted[-1] = 0.
        mask_shifted = ops.pad(mask, [[0, 0], [0, 1]])[:, 1:]
        # last_token_mask[i] = 1 iff mask[i]=1 and mask[i+1]=0.
        last_token_mask = mask * (1.0 - mask_shifted)
        # Weighted sum selects the last real token embedding.
        # Shape: (batch, hidden_dim).
        return ops.sum(
            sequence_output * ops.expand_dims(last_token_mask, axis=-1),
            axis=1,
        )

    @staticmethod
    def _mean_pooling(sequence_output, padding_mask):
        """Attention-mask-aware mean pooling over non-padding tokens."""
        mask = ops.cast(
            ops.expand_dims(padding_mask, axis=-1), sequence_output.dtype
        )
        sum_embeddings = ops.sum(sequence_output * mask, axis=1)
        sum_mask = ops.maximum(ops.sum(mask, axis=1), 1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def _l2_normalize(embeddings):
        """L2-normalize embeddings along the last axis."""
        return ops.nn.normalize(embeddings, axis=-1, order=2)

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
