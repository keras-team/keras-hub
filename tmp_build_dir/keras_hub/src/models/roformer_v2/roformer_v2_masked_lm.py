import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.masked_lm import MaskedLM
from keras_hub.src.models.roformer_v2.roformer_v2_backbone import (
    RoformerV2Backbone,
)
from keras_hub.src.models.roformer_v2.roformer_v2_masked_lm_preprocessor import (  # noqa: E501
    RoformerV2MaskedLMPreprocessor,
)


class RoformerV2MaskedLMHead(keras.layers.Layer):
    def __init__(
        self,
        vocabulary_size=None,
        token_embedding=None,
        activation=None,
        **kwargs,
    ):
        super().__init__(**kwargs, autocast=False)

        self.token_embedding = token_embedding
        self.activation = keras.activations.get(activation)

        if vocabulary_size and vocabulary_size != token_embedding.input_dim:
            raise ValueError(
                "`vocabulary_size` should match the input dimension of the "
                "of `token_embedding`. Received: "
                f"`vocabulary_size={vocabulary_size}`, "
                f"`token_embedding.input_dim={token_embedding.input_dim}`"
            )
        self.vocabulary_size = token_embedding.input_dim

    def call(self, inputs, mask_positions):
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            # On the tf backend, we need to work around an issue with dynamic
            # shape broadcasting in take_along_axis.
            x = tf.gather(inputs, mask_positions, batch_dims=1)
        else:
            # Gather the encoded tokens at the masked indices.
            mask_positions = ops.expand_dims(mask_positions, axis=-1)
            x = ops.take_along_axis(inputs, mask_positions, axis=1)

        outputs = self.token_embedding(x, reverse=True)

        # Apply a final activation.
        if self.activation is not None:
            outputs = self.activation(outputs)

        outputs = ops.cast(outputs, "float32")
        return outputs

    def get_config(self):
        config = super().get_config()
        embedding_config = None
        if self.token_embedding:
            embedding_config = keras.layers.serialize(self.token_embedding)
        config.update(
            {
                "token_embedding": embedding_config,
                "vocabulary_size": self.vocabulary_size,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config


@keras_hub_export("keras_hub.models.RoformerV2MaskedLM")
class RoformerV2MaskedLM(MaskedLM):
    """An end-to-end Roformer model for the masked language modeling task.

    This model will train RoformerV2 on a masked language modeling task.
    The model will predict labels for a number of masked tokens in the
    input data. For usage of this model with pre-trained weights, see the
    `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case inputs can be raw string features during `fit()`, `predict()`,
    and `evaluate()`. Inputs will be tokenized and dynamically masked during
    training and evaluation. This is done by default when creating the model
    with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.RoformerV2Backbone` instance.
        preprocessor: A `keras_hub.models.RoformerV2MaskedLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]

    # Pretrained language model.
    masked_lm = keras_hub.models.RoformerV2MaskedLM.from_preset(
        "roformer_v2_base_zh",
    )
    masked_lm.fit(x=features, batch_size=2)

    # Re-compile (e.g., with a new learning rate).
    masked_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        jit_compile=True,
    )
    # Access backbone programmatically (e.g., to change `trainable`).
    masked_lm.backbone.trainable = False
    # Fit again.
    masked_lm.fit(x=features, batch_size=2)
    ```

    Preprocessed integer data.
    ```python
    # Create preprocessed batch where 0 is the mask token.
    features = {
        "token_ids": np.array([[1, 2, 0, 4, 0, 6, 7, 8]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2),
        "mask_positions": np.array([[2, 4]] * 2),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 0, 0, 0]] * 2)
    }
    # Labels are the original masked values.
    labels = [[3, 5]] * 2

    masked_lm = keras_hub.models.RoformerV2MaskedLM.from_preset(
        "roformer_v2_base_zh",
        preprocessor=None,
    )
    masked_lm.fit(x=features, y=labels, batch_size=2)
    ```
    """

    backbone_cls = RoformerV2Backbone
    preprocessor_cls = RoformerV2MaskedLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.masked_lm_head = RoformerV2MaskedLMHead(
            vocabulary_size=backbone.vocabulary_size,
            token_embedding=backbone.token_embedding,
            name="mlm_head",
        )

        # === Functional Model ===
        inputs = {
            **backbone.input,
            "mask_positions": keras.Input(
                shape=(None,), dtype="int32", name="mask_positions"
            ),
        }
        backbone_outputs = backbone(backbone.input)
        outputs = self.masked_lm_head(
            backbone_outputs, inputs["mask_positions"]
        )
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
