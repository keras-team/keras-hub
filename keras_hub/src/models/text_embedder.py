import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.TextEmbedder")
class TextEmbedder(Task):
    """Base class for all text embedding tasks.

    `TextEmbedder` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    generating fixed-size sentence embeddings from variable-length text inputs.

    All `TextEmbedder` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Example:
    ```python
    # Load a sentence-transformers model.
    embedder = keras_hub.models.TextEmbedder.from_preset(
        "all_minilm_l6_v2_en",
    )

    # Semantic search.
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Mars is often referred to as the Red Planet.",
        "Venus is often called Earth's twin.",
    ]
    q_emb = embedder.encode_query(query)
    d_embs = embedder.encode_documents(documents)
    sims = embedder.similarity(q_emb, d_embs)
    print("Best match:", documents[sims.argmax()])
    ```
    """

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        """Configures the `TextEmbedder` task for training.

        The `TextEmbedder` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.CosineSimilarity` loss will be
                applied for the embedding task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.CosineSimilarity` will be
                applied to track the similarity of the model during training.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        if loss == "auto":
            loss = keras.losses.CosineSimilarity()
        if metrics == "auto":
            metrics = [keras.metrics.CosineSimilarity()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )

    def encode_query(self, query, **kwargs):
        """Encode a query string or list of queries into embeddings.

        This is a convenience method that wraps `predict()` for single
        query or batch of queries. The output embeddings are suitable for
        computing similarity against document embeddings.

        Args:
            query: A string or list of strings to encode.
            **kwargs: Additional keyword arguments passed to `predict()`.

        Returns:
            A numpy array of shape `(batch_size, embedding_dim)`.
        """
        if isinstance(query, str):
            query = [query]
        return self.predict(query, **kwargs)

    def encode_documents(self, documents, **kwargs):
        """Encode a list of documents into embeddings.

        This is a convenience method that wraps `predict()` for a batch
        of documents. The output embeddings are suitable for computing
        similarity against query embeddings.

        Args:
            documents: A list of strings to encode.
            **kwargs: Additional keyword arguments passed to `predict()`.

        Returns:
            A numpy array of shape `(batch_size, embedding_dim)`.
        """
        return self.predict(documents, **kwargs)

    def similarity(self, query_embeddings, document_embeddings):
        """Compute similarity between query and document embeddings.

        Computes the dot product between query and document embeddings.
        When embeddings are L2 normalized (the default for
        `BertTextEmbedder`), this is equivalent to cosine similarity.
        Returns a similarity matrix of shape
        `(num_queries, num_documents)`.

        Args:
            query_embeddings: An array or tensor of shape
                `(num_queries, embedding_dim)`.
            document_embeddings: An array or tensor of shape
                `(num_documents, embedding_dim)`.

        Returns:
            An array of shape `(num_queries, num_documents)` containing
            similarity scores.
        """
        query_tensor = ops.convert_to_tensor(query_embeddings)
        document_tensor = ops.convert_to_tensor(document_embeddings)
        return ops.convert_to_numpy(
            ops.matmul(query_tensor, ops.transpose(document_tensor))
        )
