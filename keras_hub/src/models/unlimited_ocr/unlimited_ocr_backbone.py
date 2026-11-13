import keras
from keras.layers import ReversibleEmbedding
from keras.layers import RMSNormalization

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_decoder import (
    UnlimitedOCRTransformerDecoder,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRImageFeatureExtractor,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRImageTextEmbeddingMerger,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_vision_encoder import (
    UnlimitedOCRVisionEncoder,
)


def _apply_dtype(vision_encoder, dtype):
    """Returns `vision_encoder` with the dtype policy of the backbone."""
    policy = keras.dtype_policies.get(dtype)
    serialized_policy = keras.dtype_policies.serialize(policy)
    if (
        keras.dtype_policies.serialize(vision_encoder.dtype_policy)
        == serialized_policy
    ):
        return vision_encoder
    if vision_encoder.built:
        raise ValueError(
            "The `vision_encoder` is built and has another dtype policy than "
            "the backbone. Create the `vision_encoder` with the same `dtype` "
            f"as the backbone. Received: dtype={dtype}, "
            f"vision_encoder.dtype_policy={vision_encoder.dtype_policy}"
        )
    config = vision_encoder.get_config()
    config["dtype"] = serialized_policy
    return UnlimitedOCRVisionEncoder.from_config(config)


@keras_hub_export("keras_hub.models.UnlimitedOCRBackbone")
class UnlimitedOCRBackbone(Backbone):
    """The Unlimited-OCR vision language backbone.

    The backbone embeds the tokens, replaces the embeddings of the image
    placeholder tokens with image features, and runs a mixture of experts
    transformer decoder. The image features come from a SAM encoder followed
    by a CLIP encoder and a linear projector. The backbone outputs the final
    hidden states of every token, not the logits. For text generation, use
    `keras_hub.models.UnlimitedOCRCausalLM`.

    Each image has one global view and up to 32 local tiles. The global view
    has a fixed size and the tiles have another fixed size. Local features
    come first in the feature sequence of an image. Then the global features
    and one view separator follow. A learned newline embedding ends every row
    of features.

    The three image inputs have the same batch size as the token inputs.
    `pixel_values` holds one global view for each image of a sample, and
    `patches_grid` holds one `(columns, rows)` pair for each image.
    `pixel_values_local` holds the tiles of all images of a sample, one
    after the other. The tiles of one image are in row-major order. An
    image with no tiles has the grid `(0, 0)`. A sample with fewer images
    than the other samples has padding images with zero pixels and the grid
    `(0, 0)`. The padding follows the real images. The placeholder tokens of
    a sample select its images in order.

    The default constructor gives a randomly initialized model. To load a
    preset architecture and its weights, use `from_preset()`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_query_heads: int. The number of attention heads.
        hidden_dim: int. The size of the hidden state.
        intermediate_dim: int. The intermediate size of the dense MLP in the
            layers of type `"dense"`.
        moe_intermediate_dim: int. The intermediate size of one routed
            expert.
        num_shared_experts: int. The number of shared experts in every sparse
            layer.
        num_experts: int. The number of routed experts in every sparse layer.
        top_k: int. The number of routed experts for each token.
        vision_encoder: A `keras_hub.models.UnlimitedOCRVisionEncoder`
            instance. If `dtype` is set and differs from the dtype of the
            vision encoder, the backbone rebuilds an unbuilt vision encoder
            with `dtype`. The backbone raises an error for a built one.
        mlp_layer_types: list of strings or `None`. The MLP type of each
            layer, either `"dense"` or `"sparse"`. A `"sparse"` layer uses a
            mixture of experts. The list has `num_layers` entries. If `None`,
            the first layer is `"dense"` and the other layers are
            `"sparse"`. Defaults to `None`.
        layer_types: list of strings or `None`. The attention type of each
            layer, either `"reference_sliding_attention"` or
            `"full_attention"`. A `"full_attention"` layer always uses full
            causal attention. A `"reference_sliding_attention"` layer uses
            the R-SWA during cached decoding. The list has `num_layers`
            entries. If `None`, every layer is
            `"reference_sliding_attention"`. Defaults to `None`.
        num_key_value_heads: int or `None`. The number of key and value
            heads. It must divide `num_query_heads`. If `None`, it equals
            `num_query_heads`. Defaults to `None`.
        routed_scaling_factor: float. The factor that scales the routing
            weights. Defaults to `1.0`.
        rope_max_wavelength: int. The maximum wavelength of the rotary
            embedding. Defaults to `10000`.
        layer_norm_epsilon: float. The epsilon of the RMS norm layers.
            Defaults to `1e-6`.
        sliding_window: int. The decode window of the reference sliding
            window attention. Defaults to `128`.
        image_token_id: int. The id of the image placeholder token. Defaults
            to `128815`.
        global_image_size: int. The side of the global view in pixels.
            Defaults to `1024`.
        local_image_size: int. The side of a local tile in pixels. Defaults
            to `640`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model computations and weights. The router, the softmax
            and the normalization layers compute in float32.

    Example:
    ```python
    vision_encoder = keras_hub.models.UnlimitedOCRVisionEncoder(
        sam_hidden_dim=16,
        sam_num_layers=2,
        sam_num_heads=2,
        sam_mlp_dim=32,
        sam_output_channels=8,
        sam_patch_size=8,
        sam_image_size=128,
        sam_window_size=4,
        sam_global_attn_indexes=(1,),
        sam_downsample_channels=(12, 16),
        clip_hidden_dim=16,
        clip_num_layers=2,
        clip_num_heads=2,
        clip_intermediate_dim=32,
        clip_image_size=48,
        clip_patch_size=16,
    )
    model = keras_hub.models.UnlimitedOCRBackbone(
        vocabulary_size=100,
        num_layers=2,
        num_query_heads=2,
        hidden_dim=16,
        intermediate_dim=32,
        moe_intermediate_dim=8,
        num_shared_experts=2,
        num_experts=4,
        top_k=2,
        vision_encoder=vision_encoder,
        image_token_id=99,
        global_image_size=128,
        local_image_size=80,
    )
    # The 4x4 global view needs 21 placeholders: 4 rows of 4 tokens and a
    # newline, plus one separator. This short input has only 7.
    model(
        {
            "token_ids": np.array([[1, 99, 99, 99, 99, 99, 99, 99, 2]]),
            "padding_mask": np.ones((1, 9), dtype="int32"),
            "pixel_values": np.zeros((1, 1, 128, 128, 3), dtype="float32"),
            "pixel_values_local": np.zeros((1, 0, 80, 80, 3), dtype="float32"),
            "patches_grid": np.zeros((1, 1, 2), dtype="int32"),
        }
    )
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        hidden_dim,
        intermediate_dim,
        moe_intermediate_dim,
        num_shared_experts,
        num_experts,
        top_k,
        vision_encoder,
        mlp_layer_types=None,
        layer_types=None,
        num_key_value_heads=None,
        routed_scaling_factor=1.0,
        rope_max_wavelength=10000,
        layer_norm_epsilon=1e-6,
        sliding_window=128,
        image_token_id=128815,
        global_image_size=1024,
        local_image_size=640,
        dtype=None,
        **kwargs,
    ):
        mlp_types = mlp_layer_types
        if mlp_types is None:
            mlp_types = ["dense"] + ["sparse"] * (num_layers - 1)
        if len(mlp_types) != num_layers or not set(mlp_types) <= {
            "dense",
            "sparse",
        }:
            raise ValueError(
                "`mlp_layer_types` must have one `'dense'` or `'sparse'` "
                f"entry for each of the {num_layers} layers. "
                f"Received: mlp_layer_types={mlp_layer_types}"
            )
        attention_types = layer_types
        if attention_types is None:
            attention_types = ["reference_sliding_attention"] * num_layers
        if len(attention_types) != num_layers or not set(attention_types) <= {
            "reference_sliding_attention",
            "full_attention",
        }:
            raise ValueError(
                "`layer_types` must have one `'reference_sliding_attention'` "
                f"or `'full_attention'` entry for each of the {num_layers} "
                f"layers. Received: layer_types={layer_types}"
            )

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = UnlimitedOCRTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                moe_intermediate_dim=moe_intermediate_dim,
                num_shared_experts=num_shared_experts,
                num_experts=num_experts,
                top_k=top_k,
                sparse=mlp_types[i] == "sparse",
                routed_scaling_factor=routed_scaling_factor,
                rope_max_wavelength=rope_max_wavelength,
                layer_norm_epsilon=layer_norm_epsilon,
                sliding_window=(
                    sliding_window
                    if attention_types[i] == "reference_sliding_attention"
                    else None
                ),
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )
        # `dtype` must reach the vision tower, which the caller creates.
        if dtype is not None:
            vision_encoder = _apply_dtype(vision_encoder, dtype)
        self.vision_encoder = vision_encoder
        self.multimodal_projector = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
            dtype=dtype,
            name="multimodal_projector",
        )
        self.image_feature_extractor = UnlimitedOCRImageFeatureExtractor(
            vision_encoder,
            self.multimodal_projector,
            dtype=dtype,
            name="image_feature_extractor",
        )
        self.image_text_embedding_merger = UnlimitedOCRImageTextEmbeddingMerger(
            hidden_dim=hidden_dim,
            image_token_id=image_token_id,
            dtype=dtype,
            name="image_text_embedding_merger",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        pixel_values_input = keras.Input(
            shape=(None, global_image_size, global_image_size, 3),
            name="pixel_values",
        )
        pixel_values_local_input = keras.Input(
            shape=(None, local_image_size, local_image_size, 3),
            name="pixel_values_local",
        )
        patches_grid_input = keras.Input(
            shape=(None, 2), dtype="int32", name="patches_grid"
        )

        x = self.token_embedding(token_id_input)
        global_features, local_features = self.image_feature_extractor(
            pixel_values_input, pixel_values_local_input
        )
        x = self.image_text_embedding_merger(
            token_id_input,
            x,
            global_features,
            local_features,
            patches_grid_input,
        )
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
                "pixel_values": pixel_values_input,
                "pixel_values_local": pixel_values_local_input,
                "patches_grid": patches_grid_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = (
            num_query_heads
            if num_key_value_heads is None
            else num_key_value_heads
        )
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.moe_intermediate_dim = moe_intermediate_dim
        self.num_shared_experts = num_shared_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.mlp_layer_types = mlp_layer_types
        self.layer_types = layer_types
        self.routed_scaling_factor = routed_scaling_factor
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.sliding_window = sliding_window
        self.image_token_id = image_token_id
        self.global_image_size = global_image_size
        self.local_image_size = local_image_size
        self.head_dim = hidden_dim // num_query_heads

    def get_config(self):
        config = super().get_config()
        # Without `build_config`, the vision encoder builds inside the
        # backbone. Its sublayers then get their dtype policy by the path of
        # the backbone, as for a freshly made backbone.
        vision_encoder_config = keras.layers.serialize(self.vision_encoder)
        vision_encoder_config.pop("build_config", None)
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "num_shared_experts": self.num_shared_experts,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "mlp_layer_types": self.mlp_layer_types,
                "layer_types": self.layer_types,
                "routed_scaling_factor": self.routed_scaling_factor,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "sliding_window": self.sliding_window,
                "image_token_id": self.image_token_id,
                "global_image_size": self.global_image_size,
                "local_image_size": self.local_image_size,
                "vision_encoder": vision_encoder_config,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["vision_encoder"] = keras.layers.deserialize(
            config["vision_encoder"]
        )
        return super().from_config(config)
