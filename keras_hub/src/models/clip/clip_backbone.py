import math

from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


class CLIPVisionPooler(layers.Layer):
    """The vision pooler layer of CLIP.

    `CLIPVisionPooler` will extracts the first token (index `0`) from the
    sequence of the vision embeddings as the pooled outputs.

    Call arguments:
        vision_embeddings: A tensor of shape
            `(batch_size, sequence_length, hidden_dim)`.
    """

    def call(self, vision_embeddings):
        return vision_embeddings[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class CLIPTextPooler(layers.Layer):
    """The text pooler layer of CLIP.

    `CLIPTextPooler` extracts the text embeddings at the positions of EOS tokens
    as the pooled outputs.

    Call arguments:
        text_embeddings: A tensor of shape
            `(batch_size, sequence_length, hidden_dim)`.
        token_ids: A tensor of shape `(batch_size, max_tokens)`, used to
            identify the positions of EOS tokens.
    """

    def call(self, text_embeddings, token_ids):
        # `keepdims` is not supported in `keras<=3.1`.
        eos_index = ops.argmax(token_ids, axis=-1)
        eos_index = ops.expand_dims(eos_index, axis=-1)
        eos_index = ops.expand_dims(eos_index, axis=-1)
        pooled_outputs = ops.take_along_axis(text_embeddings, eos_index, axis=1)
        return ops.squeeze(pooled_outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class CLIPHead(layers.Layer):
    """The head layer of CLIP.

    `CLIPHead` takes `vision_embedding` and `text_embedding` as inputs to
    compute the corresponding logits. Both embeddings are L2 normalized and used
    to compute pairwise cosine similarity. The resulting logits are then scaled
    by a learnable `logit_scale` parameter.

    Call arguments:
        vision_embedding: A tensor of shape `(batch_size, hidden_dim)`.
        text_embedding: A tensor of shape `(batch_size, hidden_dim)`.
    """

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            shape=(),
            initializer=lambda *a, **kw: math.log(1 / 0.07),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_scale",
        )

    def call(self, vision_embedding, text_embedding):
        normalized_vision_embedding = ops.sqrt(
            ops.sum(ops.power(vision_embedding, 2), axis=-1, keepdims=True)
        )
        normalized_text_embedding = ops.sqrt(
            ops.sum(ops.power(text_embedding, 2), axis=-1, keepdims=True)
        )
        vision_embedding = vision_embedding / normalized_vision_embedding
        text_embedding = text_embedding / normalized_text_embedding
        logit_scale = ops.exp(self.logit_scale)
        text_logits = (
            ops.matmul(
                text_embedding,
                ops.transpose(vision_embedding),
            )
            * logit_scale
        )
        vision_logits = ops.transpose(text_logits)
        return vision_logits, text_logits

    def compute_output_shape(
        self, vision_embedding_shape, text_embedding_shape
    ):
        vision_logits_shape = (
            vision_embedding_shape[0],
            text_embedding_shape[0],
        )
        text_logits_shape = (
            text_embedding_shape[0],
            vision_embedding_shape[0],
        )
        return vision_logits_shape, text_logits_shape


@keras_hub_export("keras_hub.models.CLIPBackbone")
class CLIPBackbone(Backbone):
    """CLIP core network with hyperparameters.

    This backbone implements the base architecture for Contrastive
    Language-Image Pretraining (CLIP) model. It includes a vision and text
    encoders and the corresponding projection layers. This backbone will output
    the final logit scores corresponding to each image and token input. These
    values are cosine similarities between the corresponding image and text
    features.

    The default constructor gives a fully customizable, randomly initialized
    CLIP model with any number of layers, heads, and embedding dimensions. To
    load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vision_encoder: The CLIP vision encoder for encoding the input images.
        text_encoder: The CLIP text encoder for encoding the input tokens.
        projection_dim: int. The size of the projection layer.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "images": np.ones(shape=(1, 224, 224, 3), dtype="float32"),
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    }

    # Pretrained CLIP model.
    model = keras_hub.models.CLIPBackbone.from_preset("clip_vit_base_patch32")
    model(input_data)

    # Randomly initialized CLIP model with custom config.
    vision_encoder = keras_hub.models.CLIPVisionEncoder(
        patch_size=32,
        hidden_dim=768,
        num_layers=8,
        num_heads=8,
        intermediate_dim=2048,
        image_shape=(384, 384, 3),
    )
    text_encoder = keras_hub.models.CLIPTextEncoder(
        vocabulary_size=49408,
        embedding_dim=768,
        hidden_dim=768,
        num_layers=8,
        num_heads=8,
        intermediate_dim=2048,
    )
    model = keras_hub.models.CLIPBackbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        projection_dim=256,
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
        self.vision_pooler = CLIPVisionPooler(dtype=dtype, name="vision_pooler")
        self.text_pooler = CLIPTextPooler(dtype=dtype, name="text_pooler")
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
        self.clip_head = CLIPHead(dtype=dtype, name="clip_head")

        # === Functional Model ===
        image_input = layers.Input(
            shape=self.vision_encoder.image_shape, name="images"
        )
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        vision_embeddings = self.get_vision_embeddings(image_input)
        text_embeddings = self.get_text_embeddings(token_id_input)
        vision_logits, text_logits = self.clip_head(
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
        vision_outputs = self.vision_pooler(vision_outputs)
        return self.vision_projection(vision_outputs)

    def get_text_embeddings(self, token_ids):
        """Get the embeddings from the text encoder.

        Args:
            token_ids: The input int tensor for the text encoder.

        Returns:
            The output embeddings obtained by applying projection layer to the
            pooled output of the text encoder.
        """
        text_outputs = self.text_encoder({"token_ids": token_ids})
        text_outputs = self.text_pooler(text_outputs, token_ids)
        return self.text_projection(text_outputs)

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
