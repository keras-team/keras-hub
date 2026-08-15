import keras
from keras import layers
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.smolvlm2.smolvlm2_layers import SmolVLM2Connector
from keras_hub.src.models.smolvlm2.smolvlm2_layers import SmolVLM2DecoderBlock
from keras_hub.src.models.smolvlm2.smolvlm2_layers import (
    SmolVLM2InterleaveEmbeddings,
)
from keras_hub.src.models.smolvlm2.smolvlm2_vision_encoder import (
    SmolVLM2VisionEncoder,
)


@keras_hub_export("keras_hub.models.SmolVLM2Backbone")
class SmolVLM2Backbone(Backbone):
    """SmolVLM2 multimodal backbone.

    This backbone implements the SmolVLM2 architecture, combining a
    SigLIP-based vision encoder, a pixel-shuffle connector, and a
    Llama-style text decoder. It can process text-only inputs or
    multimodal (image + text) inputs.

    The vision encoder extracts patch features from images, the
    connector compresses and projects them to the text decoder's
    hidden dimension, and the `interleave_embeddings` layer replaces
    positions indicated by `vision_indices` in the text token sequence
    with the projected visual embeddings.

    For text-only calls, a `__call__` override injects zero-sized
    dummy vision tensors so the functional graph receives all required
    keys. This follows the Qwen3.5/Gemma4 multimodal backbone pattern.

    Args:
        vocabulary_size: int. Size of the token vocabulary.
        image_size: int. Input image resolution (square).
        patch_size: int. Vision encoder patch size.
        vision_hidden_dim: int. Vision encoder hidden dimension.
        vision_intermediate_dim: int. Vision encoder MLP intermediate dim.
        vision_num_layers: int. Number of vision encoder blocks.
        vision_num_heads: int. Number of vision encoder attention heads.
        hidden_dim: int. Text decoder hidden dimension.
        intermediate_dim: int. Text decoder MLP intermediate dimension.
        num_layers: int. Number of text decoder blocks.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads.
        scale_factor: int. Pixel-shuffle spatial downsampling factor.
        image_token_id: int. Token ID used as a placeholder for image
            embeddings in the text sequence.
        rope_max_wavelength: float. Maximum wavelength for RoPE.
            Defaults to `10000`.
        layer_norm_epsilon: float. Epsilon for normalization layers.
            Defaults to `1e-5`.
        vision_layer_norm_epsilon: float. Epsilon for vision encoder
            LayerNorm. Defaults to `1e-6`.
        tie_word_embeddings: bool. Whether to tie the token embedding
            weights with the output projection. Defaults to `False`.
        dtype: string or `keras.mixed_precision.DTypePolicy`.

    Examples:
    ```python
    # Text-only forward pass (vision inputs injected automatically).
    backbone = keras_hub.models.SmolVLM2Backbone(
        vocabulary_size=49280,
        image_size=384,
        patch_size=14,
        vision_hidden_dim=1152,
        vision_intermediate_dim=4304,
        vision_num_layers=27,
        vision_num_heads=16,
        hidden_dim=2048,
        intermediate_dim=8192,
        num_layers=24,
        num_query_heads=32,
        num_key_value_heads=32,
        scale_factor=3,
        image_token_id=49190,
    )
    input_data = {
        "token_ids": np.ones((1, 12), dtype="int32"),
        "padding_mask": np.ones((1, 12), dtype="int32"),
    }
    output = backbone(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        image_size,
        patch_size,
        vision_hidden_dim,
        vision_intermediate_dim,
        vision_num_layers,
        vision_num_heads,
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        scale_factor,
        image_token_id,
        rope_max_wavelength=10000,
        layer_norm_epsilon=1e-5,
        vision_layer_norm_epsilon=1e-6,
        tie_word_embeddings=False,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        # Vision encoder.
        self.vision_encoder = SmolVLM2VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=vision_hidden_dim,
            intermediate_dim=vision_intermediate_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            layer_norm_epsilon=vision_layer_norm_epsilon,
            dtype=dtype,
            name="vision_encoder",
        )

        # Connector: pixel-shuffle + linear projection.
        self.connector = SmolVLM2Connector(
            vision_hidden_dim=vision_hidden_dim,
            text_hidden_dim=hidden_dim,
            scale_factor=scale_factor,
            dtype=dtype,
            name="connector",
        )

        # Interleave layer for merging vision into text embeddings.
        self.interleave_embeddings = SmolVLM2InterleaveEmbeddings(
            hidden_dim=hidden_dim,
            dtype=dtype,
            name="interleave_embeddings",
        )

        # Token embedding (reversible for weight tying with lm_head).
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            dtype=dtype,
            name="token_embedding",
        )

        # Decoder blocks.
        self.transformer_layers = []
        for i in range(num_layers):
            layer = SmolVLM2DecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)

        # Final normalization.
        self.layer_norm = layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        # Compute image sequence length from config.
        num_patches = (image_size // patch_size) ** 2
        image_seq_len = num_patches // (scale_factor**2)

        # === Functional Model ===
        # Vision inputs.
        pixel_values_input = keras.Input(
            shape=(image_size, image_size, 3),
            name="pixel_values",
        )
        vision_indices_input = keras.Input(
            shape=(None,), dtype="int32", name="vision_indices"
        )

        # Text inputs.
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Vision: encoder → connector → interleave into text.
        img_embeddings = self.vision_encoder(
            {"pixel_values": pixel_values_input}
        )
        img_embeddings = self.connector(img_embeddings)

        # Text embeddings.
        text_embeddings = self.token_embedding(token_id_input)

        # Merge vision into text at vision_indices positions.
        x = self.interleave_embeddings(
            image_embeddings=img_embeddings,
            text_embeddings=text_embeddings,
            vision_indices=vision_indices_input,
        )

        # Decoder blocks.
        for decoder_block in self.transformer_layers:
            x = decoder_block(x, decoder_padding_mask=padding_mask_input)

        sequence_output = self.layer_norm(x)

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "pixel_values": pixel_values_input,
            "vision_indices": vision_indices_input,
        }

        super().__init__(
            inputs=inputs,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_intermediate_dim = vision_intermediate_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.scale_factor = scale_factor
        self.image_token_id = image_token_id
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.vision_layer_norm_epsilon = vision_layer_norm_epsilon
        self.tie_word_embeddings = tie_word_embeddings
        self.image_sequence_length = image_seq_len

    def __call__(self, inputs, *args, **kwargs):
        """Override to inject default empty vision inputs for text-only calls.

        When the backbone receives text-only inputs (no `pixel_values`
        or `vision_indices`), this injects zero-sized dummy tensors so
        the functional graph receives all required keys. This follows
        the Qwen3.5/Gemma4 multimodal backbone pattern.
        """
        if isinstance(inputs, dict):
            inputs = dict(inputs)  # shallow copy to avoid mutation
            batch_size = ops.shape(inputs["token_ids"])[0]
            if "pixel_values" not in inputs:
                inputs["pixel_values"] = ops.zeros(
                    (
                        batch_size,
                        self.image_size,
                        self.image_size,
                        3,
                    ),
                )
            if "vision_indices" not in inputs:
                inputs["vision_indices"] = ops.zeros(
                    (batch_size, 0), dtype="int32"
                )
        return super().__call__(inputs, *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_intermediate_dim": self.vision_intermediate_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "scale_factor": self.scale_factor,
                "image_token_id": self.image_token_id,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "vision_layer_norm_epsilon": (self.vision_layer_norm_epsilon),
                "tie_word_embeddings": self.tie_word_embeddings,
            }
        )
        return config
