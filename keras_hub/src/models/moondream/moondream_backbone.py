import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.phi3.phi3_decoder import Phi3Decoder
from keras_hub.src.models.phi3.phi3_layernorm import Phi3LayerNorm
from keras_hub.src.models.siglip.siglip_layers import SigLIPEncoderLayer
from keras_hub.src.models.siglip.siglip_layers import SigLIPVisionEmbedding


def _moondream_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.MoondreamBackbone")
class MoondreamBackbone(Backbone):
    """Moondream core network with hyperparameters.

    This backbone implements the Moondream vision-language model architecture.
    It contains a SigLIP-style vision encoder that converts images to patch
    embeddings, a linear projection layer that maps vision embeddings into the
    text decoder's hidden dimension, and a Phi-style causal text decoder. Image
    patch embeddings are prepended to the text token embeddings before being
    passed through the decoder stack.

    For a higher-level object for image-to-text generation, see
    `keras_hub.models.MoondreamCausalLM`.

    The default constructor gives a fully customizable, randomly initialized
    Moondream model with any number of vision and text layers, heads, and
    embedding dimensions. To load preset architectures and weights, use the
    `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        image_size: int. The resolution of the input image in both height and
            width. Note: input images must be square.
        vision_patch_size: int. The size of each square patch in the image.
        vision_num_layers: int. The number of transformer layers in the vision
            encoder.
        vision_num_heads: int. The number of attention heads in each vision
            encoder transformer layer.
        vision_hidden_dim: int. The hidden state dimension of the vision
            encoder transformer.
        vision_intermediate_dim: int. The output dimension of the first Dense
            layer in the feedforward network of each vision encoder layer.
        vision_layer_norm_epsilon: float. The epsilon value for layer
            normalization in the vision encoder. Defaults to `1e-6`.
        projection_dim: int. The output dimension of the vision-to-text
            projection layer. Should equal `text_hidden_dim`.
        text_num_layers: int. The number of transformer decoder layers in the
            text decoder.
        text_hidden_dim: int. The hidden state dimension of the text decoder.
        text_intermediate_dim: int. The output dimension of the first Dense
            layer in the feedforward network of each text decoder layer.
        text_num_query_heads: int. The number of query attention heads in each
            text decoder layer.
        text_num_key_value_heads: int. The number of key and value attention
            heads in each text decoder layer.
        text_layer_norm_epsilon: float. The epsilon value for layer
            normalization in the text decoder. Defaults to `1e-5`.
        text_dropout: float. Dropout probability for the text decoder.
            Defaults to `0.0`.
        text_max_sequence_length: int. The maximum sequence length the text
            decoder supports. Defaults to `2048`.
        text_rope_max_wavelength: int. The maximum angular wavelength for
            rotary position embeddings in the text decoder. Defaults to
            `10000`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights. Note that some
            computations, such as softmax and layer normalization, will always
            be done at float32 precision regardless of dtype.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Randomly initialized Moondream backbone with a small config.
    backbone = keras_hub.models.MoondreamBackbone(
        vocabulary_size=51200,
        image_size=378,
        vision_patch_size=14,
        vision_num_layers=2,
        vision_num_heads=2,
        vision_hidden_dim=8,
        vision_intermediate_dim=16,
        projection_dim=8,
        text_num_layers=2,
        text_hidden_dim=8,
        text_intermediate_dim=16,
        text_num_query_heads=2,
        text_num_key_value_heads=1,
    )
    input_data = {
        "images": np.random.uniform(size=(2, 378, 378, 3)).astype("float32"),
        "token_ids": np.random.randint(0, 51200, (2, 16)),
        "padding_mask": np.ones((2, 16), dtype="int32"),
    }
    output = backbone(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        image_size,
        vision_patch_size,
        vision_num_layers,
        vision_num_heads,
        vision_hidden_dim,
        vision_intermediate_dim,
        vision_layer_norm_epsilon=1e-6,
        projection_dim=2048,
        text_num_layers=24,
        text_hidden_dim=2048,
        text_intermediate_dim=8192,
        text_num_query_heads=32,
        text_num_key_value_heads=32,
        text_layer_norm_epsilon=1e-5,
        text_dropout=0.0,
        text_max_sequence_length=2048,
        text_rope_max_wavelength=10000,
        dtype=None,
        **kwargs,
    ):
        # ============================================================
        # === Vision encoder layers (SigLIP-style) ===
        # ============================================================
        self.vision_embedding = SigLIPVisionEmbedding(
            hidden_dim=vision_hidden_dim,
            patch_size=vision_patch_size,
            image_size=image_size,
            dtype=dtype,
            name="vision_embedding",
        )
        self.vision_encoder_layers = []
        for i in range(vision_num_layers):
            layer = SigLIPEncoderLayer(
                num_heads=vision_num_heads,
                hidden_dim=vision_hidden_dim,
                intermediate_dim=vision_intermediate_dim,
                intermediate_activation="gelu_approximate",
                layer_norm_epsilon=vision_layer_norm_epsilon,
                dtype=dtype,
                name=f"vision_encoder_layer_{i}",
            )
            self.vision_encoder_layers.append(layer)
        self.vision_layer_norm = keras.layers.LayerNormalization(
            epsilon=vision_layer_norm_epsilon,
            dtype=dtype,
            name="vision_layer_norm",
        )

        # ============================================================
        # === Vision-to-text projection ===
        # ============================================================
        self.vision_projection = keras.layers.Dense(
            projection_dim,
            use_bias=True,
            dtype=dtype,
            name="vision_projection",
        )

        # ============================================================
        # === Text decoder layers (Phi-style) ===
        # ============================================================
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=text_hidden_dim,
            tie_weights=False,
            embeddings_initializer=_moondream_kernel_initializer(stddev=0.02),
            dtype=dtype,
            name="token_embedding",
        )
        self.text_transformer_layers = []
        for i in range(text_num_layers):
            layer = Phi3Decoder(
                hidden_dim=text_hidden_dim,
                intermediate_dim=text_intermediate_dim,
                num_query_heads=text_num_query_heads,
                num_key_value_heads=text_num_key_value_heads,
                layer_norm_epsilon=text_layer_norm_epsilon,
                activation="silu",
                kernel_initializer=_moondream_kernel_initializer(stddev=0.02),
                dropout=text_dropout,
                max_sequence_length=text_max_sequence_length,
                rope_max_wavelength=text_rope_max_wavelength,
                dtype=dtype,
                name=f"text_transformer_layer_{i}",
            )
            self.text_transformer_layers.append(layer)
        self.text_layer_norm = Phi3LayerNorm(
            epsilon=text_layer_norm_epsilon,
            dtype=dtype,
            name="text_layer_norm",
        )

        # ============================================================
        # === Functional Model ===
        # ============================================================
        image_input = keras.Input(
            shape=(image_size, image_size, 3), name="images"
        )
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Vision encoder: image -> patch embeddings -> projected embeddings.
        vision_x = self.vision_embedding(image_input)
        for vision_layer in self.vision_encoder_layers:
            vision_x = vision_layer(vision_x)
        vision_x = self.vision_layer_norm(vision_x)
        image_embeddings = self.vision_projection(vision_x)

        # Text token embeddings.
        text_embeddings = self.token_embedding(token_id_input)

        # Prepend image embeddings to text embeddings along sequence axis.
        x = ops.concatenate([image_embeddings, text_embeddings], axis=1)

        # Build combined attention mask: image patches are always attended to.
        batch_size = ops.shape(image_input)[0]
        num_patches = ops.shape(image_embeddings)[1]
        image_mask = ops.ones((batch_size, num_patches), dtype="int32")
        combined_mask = ops.concatenate(
            [image_mask, padding_mask_input], axis=1
        )

        # Text decoder: combined embeddings -> sequence output.
        for text_layer in self.text_transformer_layers:
            x = text_layer(x, decoder_padding_mask=combined_mask)
        sequence_output = self.text_layer_norm(x)

        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # ============================================================
        # === Config ===
        # ============================================================
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.vision_patch_size = vision_patch_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_intermediate_dim = vision_intermediate_dim
        self.vision_layer_norm_epsilon = vision_layer_norm_epsilon
        self.projection_dim = projection_dim
        self.text_num_layers = text_num_layers
        self.text_hidden_dim = text_hidden_dim
        self.text_intermediate_dim = text_intermediate_dim
        self.text_num_query_heads = text_num_query_heads
        self.text_num_key_value_heads = text_num_key_value_heads
        self.text_layer_norm_epsilon = text_layer_norm_epsilon
        self.text_dropout = text_dropout
        self.text_max_sequence_length = text_max_sequence_length
        self.text_rope_max_wavelength = text_rope_max_wavelength
        # Derived: number of image patch tokens prepended to every sequence.
        self.image_sequence_length = (image_size // vision_patch_size) ** 2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "vision_patch_size": self.vision_patch_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_intermediate_dim": self.vision_intermediate_dim,
                "vision_layer_norm_epsilon": self.vision_layer_norm_epsilon,
                "projection_dim": self.projection_dim,
                "text_num_layers": self.text_num_layers,
                "text_hidden_dim": self.text_hidden_dim,
                "text_intermediate_dim": self.text_intermediate_dim,
                "text_num_query_heads": self.text_num_query_heads,
                "text_num_key_value_heads": self.text_num_key_value_heads,
                "text_layer_norm_epsilon": self.text_layer_norm_epsilon,
                "text_dropout": self.text_dropout,
                "text_max_sequence_length": self.text_max_sequence_length,
                "text_rope_max_wavelength": self.text_rope_max_wavelength,
            }
        )
        return config
