"""MetaCLIP 2 backbone model implementation."""

from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.metaclip_2.metaclip_2_layers import MetaCLIP2Head
from keras_hub.src.models.metaclip_2.metaclip_2_layers import (
    MetaCLIP2TextPooler,
)
from keras_hub.src.models.metaclip_2.metaclip_2_layers import (
    MetaCLIP2VisionPooler,
)


@keras_hub_export("keras_hub.models.MetaCLIP2Backbone")
class MetaCLIP2Backbone(Backbone):
    """MetaCLIP 2 core network with hyperparameters.

    This backbone implements the base architecture for Meta's Contrastive
    Language-Image Pretraining 2 (MetaCLIP 2) model. It includes a vision and
    text encoder along with corresponding projection layers. This backbone will
    output the final logit scores corresponding to each image and token input.
    These values are cosine similarities between the corresponding image and
    text features.

    MetaCLIP 2 uses the same architecture as CLIP but is trained on a larger
    and more diverse dataset using improved data curation techniques. It uses
    `quick_gelu` activation by default.

    The default constructor gives a fully customizable, randomly initialized
    MetaCLIP 2 model with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vision_encoder: The MetaCLIP 2 vision encoder for encoding input images.
        text_encoder: The MetaCLIP 2 text encoder for encoding input tokens.
        projection_dim: int. The size of the projection layer.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done at float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "images": np.ones(shape=(1, 224, 224, 3), dtype="float32"),
        "token_ids": np.ones(shape=(1, 77), dtype="int32"),
    }

    # Pretrained MetaCLIP 2 model.
    model = keras_hub.models.MetaCLIP2Backbone.from_preset(
        "metaclip_2_vit_huge_patch14_224"
    )
    model(input_data)

    # Randomly initialized MetaCLIP 2 model with custom config.
    vision_encoder = keras_hub.models.MetaCLIP2VisionEncoder(
        patch_size=14,
        hidden_dim=1280,
        num_layers=32,
        num_heads=16,
        intermediate_dim=5120,
        image_shape=(224, 224, 3),
    )
    text_encoder = keras_hub.models.MetaCLIP2TextEncoder(
        vocabulary_size=901629,
        embedding_dim=1024,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        intermediate_dim=4096,
    )
    model = keras_hub.models.MetaCLIP2Backbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        projection_dim=1024,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        projection_dim,
        dtype=None,
        name=None,
        **kwargs,
    ):
        # === Layers ===
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_post_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, dtype=dtype, name="vision_post_layer_norm"
        )
        self.vision_pooler = MetaCLIP2VisionPooler(
            dtype=dtype, name="vision_pooler"
        )
        self.text_pooler = MetaCLIP2TextPooler(dtype=dtype, name="text_pooler")

        self.vision_projection = layers.Dense(
            projection_dim,
            use_bias=False,
            dtype=dtype,
            name="vision_projection",
        )
        self.text_projection = layers.Dense(
            projection_dim,
            use_bias=False,
            dtype=dtype,
            name="text_projection",
        )
        self.metaclip_2_head = MetaCLIP2Head(
            dtype=dtype, name="metaclip_2_head"
        )

        # === Functional Model ===
        image_input = layers.Input(
            shape=self.vision_encoder.image_shape, name="images"
        )
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        vision_embeddings = self.get_vision_embeddings(image_input)
        text_embeddings = self.get_text_embeddings(token_id_input)
        vision_logits, text_logits = self.metaclip_2_head(
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
            name=name,
            **kwargs,
        )

        # === Config ===
        self.projection_dim = projection_dim

    def get_vision_embeddings(self, images):
        """Get the embeddings from the vision encoder.

        Args:
            images: The input tensor for the vision encoder.

        Returns:
            The output embeddings obtained by applying projection layer to the
            pooled output of the vision encoder.
        """
        vision_outputs = self.vision_encoder({"images": images})
        pooled_output = self.vision_pooler(vision_outputs)
        pooled_output = self.vision_post_layer_norm(pooled_output)
        return self.vision_projection(pooled_output)

    def get_text_embeddings(self, token_ids):
        """Get the embeddings from the text encoder.

        Args:
            token_ids: The input int tensor for the text encoder.

        Returns:
            The output embeddings obtained by applying projection layer to the
            pooled output of the text encoder.
        """
        text_outputs = self.text_encoder({"token_ids": token_ids})
        pooled_output = self.text_pooler(text_outputs, token_ids)
        return self.text_projection(pooled_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": layers.serialize(self.vision_encoder),
                "text_encoder": layers.serialize(self.text_encoder),
                "projection_dim": self.projection_dim,
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
