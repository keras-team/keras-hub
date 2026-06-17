import keras

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
    q_emb = embedder.encode_text(query)
    d_embs = embedder.encode_text(documents)
    ```
    """

    def compile(
        self,
        optimizer="auto",
        loss=None,
        *,
        metrics=None,
        **kwargs,
    ):
        """Configures the `TextEmbedder` task for training.

        The `TextEmbedder` task extends the default compilation signature of
        `keras.Model.compile` with a default for `optimizer`. `loss` and
        `metrics` must be specified by the user, as there is no single
        standard loss that fits all sentence-transformer training
        scenarios.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default
                optimizer for the given model and task. See
                `keras.Model.compile` and `keras.optimizers` for more
                info on possible `optimizer` values.
            loss: A loss name or a `keras.losses.Loss` instance.
                Must be specified by the user. See `keras.Model.compile`
                and `keras.losses` for more info on possible values.
            metrics: A list of metrics to be evaluated by the model
                during training and testing. See `keras.Model.compile`
                and `keras.metrics` for more info on possible values.
            **kwargs: See `keras.Model.compile` for a full list of
                arguments supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )

    def encode_text(self, text, **kwargs):
        """Encode a string or list of strings into embeddings.

        This is a convenience method that wraps `predict()` for single
        text or batch of texts.

        Args:
            text: A string or list of strings to encode.
            **kwargs: Additional keyword arguments passed to `predict()`.

        Returns:
            Embeddings of shape `(batch_size, embedding_dim)`.
        """
        if isinstance(text, str):
            text = [text]
        return self.predict(text, **kwargs)
