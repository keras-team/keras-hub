import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.siglip.siglip_layers import SigLIPHead
from keras_hub.src.models.siglip.siglip_loss import SigLIPLoss


@keras_hub_export("keras_hub.models.SigLIPBackbone")
class SigLIPBackbone(Backbone):
    """SigCLIP core network with hyperparameters.

    This backbone implements the base architecture for the Sigmoid loss in the
    Language-Image Pre-training (SigLIP) model. Unlike standard contrastive
    learning with softmax normalization, the sigmoid loss operates solely on
    image-text pairs and does not require a global view of the pairwise
    similarities for normalization. It includes vision and text encoders. This
    backbone outputs the final logit scores corresponding to each image and
    token input.

    The default constructor gives a fully customizable, randomly initialized
    SigLIP model with any number of layers, heads, and embedding dimensions. To
    load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vision_encoder: The SigLIP vision encoder for encoding the input images.
        text_encoder: The SigLIP text encoder for encoding the input tokens.
        projection_dim: int. The size of the projection layer.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "images": np.ones(shape=(1, 224, 224, 3), dtype="float32"),
        "token_ids": np.ones(shape=(1, 64), dtype="int32"),
    }

    # Pretrained SigLIP model.
    model = keras_hub.models.SigLIPBackbone.from_preset(
        "siglip_base_patch16_224"
    )
    model(input_data)

    # Randomly initialized SigLIP model with custom config.
    vision_encoder = keras_hub.models.SigLIPVisionEncoder(
        patch_size=32,
        hidden_dim=768,
        num_layers=8,
        num_heads=8,
        intermediate_dim=2048,
        image_shape=(384, 384, 3),
    )
    text_encoder = keras_hub.models.SigLIPTextEncoder(
        vocabulary_size=32000,
        embedding_dim=768,
        hidden_dim=768,
        num_layers=8,
        num_heads=8,
        intermediate_dim=2048,
    )
    model = keras_hub.models.SigLIPBackbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.siglip_head = SigLIPHead(dtype=dtype, name="siglip_head")

        # === Functional Model ===
        image_input = layers.Input(
            shape=self.vision_encoder.image_shape, name="images"
        )
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        vision_embeddings = self.get_vision_embeddings(image_input)
        text_embeddings = self.get_text_embeddings(token_id_input)
        vision_logits, text_logits = self.siglip_head(
            vision_embeddings, text_embeddings
        )

        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
            },
            outputs={
                "vision_logits": vision_logits,
                "text_logits": text_logits,
            },
            dtype=dtype,
            **kwargs,
        )

    def compute_loss(
        self, x, y=None, y_pred=None, sample_weight=None, **kwargs
    ):
        outputs = self(x)
        text_logits = outputs["text_logits"]
        batch_size = ops.shape(text_logits)[0]
        eye = ops.eye(batch_size, dtype=text_logits.dtype)
        m1_diag1 = -ops.ones_like(text_logits) + 2 * eye
        return super().compute_loss(
            x=x,
            y=m1_diag1,
            y_pred=text_logits,
            sample_weight=sample_weight,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        metrics=None,
        **kwargs,
    ):
        """Configures the `SigLIPBackbone` task for training.

        `SigLIPBackbone` extends the default compilation signature
        of `keras.Model.compile` with defaults for `optimizer` and `loss`. To
        override these defaults, pass any value to these arguments during
        compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default
                optimizer for `SigLIPBackbone`. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer`
                values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, in which case the default loss
                computation of `SigLIPBackbone` will be applied.
                See `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `a list of metrics to be evaluated by
                the model during training and testing. Defaults to `None`.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            # Using the alternative optimizer AdamW instead of the
            # ScalingViT-Adafactor optimizer mentioned in the paper:
            # https://arxiv.org/abs/2303.15343 - C. Robustness of SigLIP
            # results.
            optimizer = keras.optimizers.AdamW(1e-3, weight_decay=1e-4)
        if loss == "auto":
            loss = SigLIPLoss()
        if metrics == "auto":
            metrics = [keras.metrics.Accuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )

    def get_vision_embeddings(self, images):
        """Get the embeddings from the vision encoder.

        Args:
            images: The input tensor for the vision encoder.

        Returns:
            The output embeddings obtained by applying projection layer to the
            pooled output of the vision encoder.
        """
        return self.vision_encoder({"images": images})

    def get_text_embeddings(self, token_ids):
        """Get the embeddings from the text encoder.

        Args:
            token_ids: The input int tensor for the text encoder.

        Returns:
            The output embeddings obtained by applying projection layer to the
            pooled output of the text encoder.
        """
        return self.text_encoder({"token_ids": token_ids})

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": layers.serialize(self.vision_encoder),
                "text_encoder": layers.serialize(self.text_encoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        # Propagate `dtype` to submodels if needed.
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["vision_encoder"]["config"]:
                config["vision_encoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["text_encoder"]["config"]:
                config["text_encoder"]["config"]["dtype"] = dtype_config

        # We expect submodels to be instantiated.
        config["vision_encoder"] = layers.deserialize(
            config["vision_encoder"], custom_objects=custom_objects
        )
        config["text_encoder"] = layers.deserialize(
            config["text_encoder"], custom_objects=custom_objects
        )
        return cls(**config)
