import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_attention import (
    compute_mrope_embeddings,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_decoder import (
    Qwen3_VLTransformerDecoder,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3_VLDeepStackInject,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3_VLInterleaveEmbeddings,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import Qwen3_VLLayerNorm


@keras_hub_export("keras_hub.models.Qwen3_VLBackbone")
class Qwen3_VLBackbone(Backbone):
    """The Qwen3-VL Transformer core architecture with hyperparameters.

    The text model is a Qwen3-style decoder with interleaved multimodal
    RoPE. When a `vision_encoder` is given, vision embeddings are scattered
    into the text embeddings, and the intermediate vision features
    ("DeepStack") are added to the hidden states after the first decoder
    layers.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_query_heads: int. The number of query attention heads.
        num_key_value_heads: int. The number of key and value attention
            heads.
        head_dim: int. The size of each attention head.
        hidden_dim: int. The size of the transformer hidden dimension.
        intermediate_dim: int. The feed-forward intermediate dimension.
        rope_max_wavelength: float. The base of the rotary frequencies.
            Defaults to `5000000.0`.
        layer_norm_epsilon: float. Epsilon of the RMSNorm layers. Defaults to
            `1e-6`.
        dropout: float. Dropout rate. Defaults to `0.0`.
        tie_word_embeddings: bool. Whether the output projection shares the
            token embedding. Defaults to `False`.
        mrope_section: list of 3 ints. The number of rotary frequencies used
            by the temporal, height and width position channels. Defaults to
            `(24, 20, 20)`.
        vision_encoder: `Qwen3_VLVisionEncoder` or `None`. When set, the
            backbone also takes `pixel_values`, `image_grid_thw` and
            `vision_indices`. Defaults to `None`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Call arguments:
        token_ids: int tensor `(batch, seq_len)`.
        padding_mask: int tensor `(batch, seq_len)`.
        position_ids: int tensor `(batch, seq_len, 3)` with the temporal,
            height and width positions. Defaults to plain 1D positions.
        pixel_values: float tensor `(batch, num_patches, temporal_patch_size,
            patch_size, patch_size, in_channels)`. Images and videos share this
            input.
        image_grid_thw: int tensor `(batch, num_items, 3)`.
        vision_indices: int tensor `(batch, num_vision_tokens)`. Indices into
            the flattened `(batch * seq_len)` sequence of the vision tokens.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        rope_max_wavelength=5000000.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        tie_word_embeddings=False,
        mrope_section=(24, 20, 20),
        vision_encoder=None,
        dtype=None,
        **kwargs,
    ):
        mrope_section = list(mrope_section)
        if (
            vision_encoder is not None
            and vision_encoder.out_hidden_size != hidden_dim
        ):
            raise ValueError(
                "`vision_encoder.out_hidden_size` must equal `hidden_dim`. "
                f"Received: {vision_encoder.out_hidden_size} and {hidden_dim}"
            )

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = Qwen3_VLTransformerDecoder(
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                mrope_section=mrope_section,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = Qwen3_VLLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        self.vision_encoder = vision_encoder
        text_only_model = vision_encoder is None
        if not text_only_model:
            self.interleave_embeddings = Qwen3_VLInterleaveEmbeddings(
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="interleave_embeddings",
            )
            self.deepstack_inject = Qwen3_VLDeepStackInject(
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="deepstack_inject",
            )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        position_ids_input = keras.Input(
            shape=(None, 3), dtype="int32", name="position_ids"
        )
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "position_ids": position_ids_input,
        }

        x = self.token_embedding(token_id_input)
        deepstack_features = []
        if not text_only_model:
            pixel_values_input = keras.Input(
                shape=(
                    None,
                    vision_encoder.temporal_patch_size,
                    vision_encoder.patch_size,
                    vision_encoder.patch_size,
                    vision_encoder.in_channels,
                ),
                name="pixel_values",
            )
            image_grid_thw_input = keras.Input(
                shape=(None, 3), dtype="int32", name="image_grid_thw"
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            inputs.update(
                {
                    "pixel_values": pixel_values_input,
                    "image_grid_thw": image_grid_thw_input,
                    "vision_indices": vision_indices_input,
                }
            )
            image_embeddings, *deepstack_features = self.vision_encoder(
                {
                    "pixel_values": pixel_values_input,
                    "image_grid_thw": image_grid_thw_input,
                }
            )
            x = self.interleave_embeddings(
                image_embeddings=image_embeddings,
                text_embeddings=x,
                vision_indices=vision_indices_input,
            )

        # The decoder layers take channel-first positions.
        position_ids = ops.transpose(position_ids_input, (0, 2, 1))
        # The rotary tables only depend on the positions, so build them once.
        rotary_embeddings = compute_mrope_embeddings(
            head_dim,
            rope_max_wavelength,
            mrope_section,
            ops.shape(x)[1],
            0,
            position_ids,
        )
        for i, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(
                x,
                decoder_padding_mask=padding_mask_input,
                position_ids=position_ids,
                rotary_embeddings=rotary_embeddings,
            )
            if i < len(deepstack_features):
                x = self.deepstack_inject(
                    x, deepstack_features[i], vision_indices_input
                )
        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs=inputs,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.mrope_section = mrope_section
        self.text_only_model = text_only_model

    def __call__(self, inputs, *args, **kwargs):
        """Fill in the inputs a caller left out.

        Missing `position_ids` default to plain 1D positions, and a
        multimodal backbone called with text only gets empty vision inputs.
        """
        if isinstance(inputs, dict):
            inputs = {k: ops.convert_to_tensor(v) for k, v in inputs.items()}
            token_ids = inputs["token_ids"]
            batch_size = ops.shape(token_ids)[0]
            if "position_ids" not in inputs:
                positions = ops.arange(ops.shape(token_ids)[1], dtype="int32")
                inputs["position_ids"] = ops.broadcast_to(
                    positions[None, :, None],
                    (batch_size, ops.shape(token_ids)[1], 3),
                )
            if not self.text_only_model:
                ve = self.vision_encoder
                if "pixel_values" not in inputs:
                    inputs["pixel_values"] = ops.zeros(
                        (
                            batch_size,
                            0,
                            ve.temporal_patch_size,
                            ve.patch_size,
                            ve.patch_size,
                            ve.in_channels,
                        ),
                    )
                if "image_grid_thw" not in inputs:
                    inputs["image_grid_thw"] = ops.zeros(
                        (batch_size, 0, 3), dtype="int32"
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
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "mrope_section": self.mrope_section,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if config["vision_encoder"] is not None:
            config["vision_encoder"] = keras.layers.deserialize(
                config["vision_encoder"]
            )
        return super().from_config(config)
