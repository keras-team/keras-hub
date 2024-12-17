import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export(
    [
        "keras_hub.models.TextClassifier",
        "keras_hub.models.Classifier",
    ]
)
class TextClassifier(Task):
    """Base class for all classification tasks.

    `TextClassifier` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    sequence classification. `TextClassifier` tasks take an additional
    `num_classes` argument, controlling the number of predicted output classes.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.

    All `TextClassifier` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Some, but not all, classification presets include classification head
    weights in a `task.weights.h5` file. For these presets, you can omit passing
    `num_classes` to restore the saved classification head. For all presets, if
    `num_classes` is passed as a kwarg to `from_preset()`, the classification
    head will be randomly initialized.

    Example:
    ```python
    # Load a BERT classifier with pre-trained weights.
    classifier = keras_hub.models.TextClassifier.from_preset(
        "bert_base_en",
        num_classes=2,
    )
    # Fine-tune on IMDb movie reviews (or any dataset).
    imdb_train, imdb_test = tfds.load(
        "imdb_reviews",
        split=["train", "test"],
        as_supervised=True,
        batch_size=16,
    )
    classifier.fit(imdb_train, validation_data=imdb_test)
    # Predict two new examples.
    classifier.predict(["What an amazing movie!", "A total waste of my time."])
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
        """Configures the `TextClassifier` task for training.

        The `TextClassifier` task extends the default compilation signature of
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
                `keras.losses.SparseCategoricalCrossentropy` loss will be
                applied for the classification task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.SparseCategoricalAccuracy` will be
                applied to track the accuracy of the model during training.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        if loss == "auto":
            activation = getattr(self, "activation", None)
            activation = keras.activations.get(activation)
            from_logits = activation != keras.activations.softmax
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits)
        if metrics == "auto":
            metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )
