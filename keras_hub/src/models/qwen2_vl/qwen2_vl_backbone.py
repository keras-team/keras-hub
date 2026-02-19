import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.models.qwen2_vl.qwen2_vl_decoder import (
    Qwen2VLTransformerDecoder,
)


def _qwen2_vl_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Qwen2VLBackbone")
class Qwen2VLBackbone(Backbone):
    """Qwen2-VL core network with optional vision encoder."""

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        mrope_section,
        rope_max_wavelength=10000,
        layer_norm_epsilon=1e-6,
        dropout=0,
        tie_word_embeddings=True,
        vision_encoder=None,
        dtype=None,
        **kwargs,
    ):
        text_only_model = vision_encoder is None
        head_dim = hidden_dim // num_query_heads

        token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen2_vl_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.vision_encoder = vision_encoder
        self._vision_interleaver = Qwen2VLInterleaveEmbeddings(
            dtype=dtype, name="vision_interleaver"
        )
        self._vision_input_flattener = None
        if vision_encoder is not None:
            self._vision_input_flattener = Qwen2VLFlattenVisionInputs(
                in_channels=vision_encoder.in_channels,
                temporal_patch_size=vision_encoder.temporal_patch_size,
                patch_size=vision_encoder.patch_size,
                dtype=dtype,
                name="vision_input_flattener",
            )

        transformer_layers = []
        for i in range(num_layers):
            layer = Qwen2VLTransformerDecoder(
                intermediate_dim=intermediate_dim,
                hidden_dim=hidden_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                mrope_section=mrope_section,
                rope_max_wavelength=rope_max_wavelength,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_qwen2_vl_kernel_initializer(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            transformer_layers.append(layer)

        layer_norm = QwenLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        mrope_position_ids_input = keras.Input(
            shape=(None, 3), dtype="int32", name="mrope_position_ids"
        )

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "mrope_position_ids": mrope_position_ids_input,
        }

        text_embeddings = token_embedding(token_id_input)

        x = text_embeddings
        if not text_only_model:
            image_input = keras.Input(
                shape=(
                    None,
                    vision_encoder.in_channels,
                    vision_encoder.temporal_patch_size,
                    vision_encoder.patch_size,
                    vision_encoder.patch_size,
                ),
                dtype="float32",
                name="images",
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            grid_thw_input = keras.Input(
                shape=(None, 3), dtype="int32", name="grid_thw"
            )

            inputs["images"] = image_input
            inputs["vision_indices"] = vision_indices_input
            inputs["grid_thw"] = grid_thw_input

            flat_images, flat_grid_thw = self._vision_input_flattener(
                images=image_input, grid_thw=grid_thw_input
            )
            vision_embeddings = self.vision_encoder(
                flat_images, grid_thw=flat_grid_thw
            )
            x = self._vision_interleaver(
                vision_embeddings=vision_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices_input,
            )

        position_embeddings = _compute_mrope_embeddings(
            mrope_position_ids_input,
            head_dim,
            rope_max_wavelength,
            mrope_section,
        )

        for transformer_layer in transformer_layers:
            x = transformer_layer(
                x,
                attention_mask=padding_mask_input,
                position_embeddings=position_embeddings,
            )

        sequence_output = layer_norm(x)

        super().__init__(
            inputs=inputs,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.mrope_section = mrope_section
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.text_only_model = text_only_model
        self.token_embedding = token_embedding
        self.transformer_layers = transformer_layers
        self.layer_norm = layer_norm

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
            }
        )
        if self.vision_encoder is not None:
            config["vision_encoder"] = keras.layers.serialize(
                self.vision_encoder
            )
        return config

    @classmethod
    def from_config(cls, config):
        vision_encoder = config.pop("vision_encoder", None)
        if vision_encoder is not None:
            vision_encoder = keras.layers.deserialize(vision_encoder)
        return cls(vision_encoder=vision_encoder, **config)


class Qwen2VLInterleaveEmbeddings(keras.layers.Layer):
    """Interleaves vision embeddings into text embeddings at given indices."""

    def call(self, vision_embeddings, text_embeddings, vision_indices):
        batch_size, seq_length, hidden_dim = ops.shape(text_embeddings)

        flat_text_embeddings = ops.reshape(
            text_embeddings, (batch_size * seq_length, hidden_dim)
        )
        # `vision_embeddings` is flattened as
        # `(batch * num_vision_tokens, hidden_dim)`.
        flat_vision_embeddings = vision_embeddings

        offsets = ops.multiply(
            ops.arange(batch_size, dtype="int32"), seq_length
        )
        offsets = ops.expand_dims(offsets, axis=-1)
        flat_indices = ops.reshape(vision_indices + offsets, (-1, 1))
        flat_indices = ops.cast(flat_indices, "int32")

        # `vision_indices` is padded with 0. Restore token 0 after scatter.
        zeroth_index_text_embeddings = ops.take(
            flat_text_embeddings,
            indices=ops.squeeze(offsets, axis=-1),
            axis=0,
        )
        reconstructed_embedding = ops.scatter_update(
            inputs=flat_text_embeddings,
            indices=flat_indices,
            updates=flat_vision_embeddings,
        )
        reconstructed_embedding = ops.scatter_update(
            inputs=reconstructed_embedding,
            indices=offsets,
            updates=zeroth_index_text_embeddings,
        )

        return ops.reshape(
            reconstructed_embedding, (batch_size, seq_length, hidden_dim)
        )

    def compute_output_shape(
        self,
        vision_embeddings_shape,
        text_embeddings_shape,
        vision_indices_shape,
    ):
        del vision_embeddings_shape
        del vision_indices_shape
        return text_embeddings_shape

    def compute_output_spec(
        self, vision_embeddings, text_embeddings, vision_indices
    ):
        output_shape = self.compute_output_shape(
            vision_embeddings.shape,
            text_embeddings.shape,
            vision_indices.shape,
        )
        return keras.KerasTensor(output_shape, dtype=text_embeddings.dtype)


class Qwen2VLFlattenVisionInputs(keras.layers.Layer):
    """Flattens batched vision patches and `grid_thw` to encoder format."""

    def __init__(
        self,
        in_channels,
        temporal_patch_size,
        patch_size,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size

    def call(self, images, grid_thw):
        flat_images = ops.reshape(
            images,
            (
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ),
        )
        flat_grid_thw = ops.reshape(grid_thw, (-1, 3))
        return flat_images, flat_grid_thw


def _compute_mrope_embeddings(
    mrope_position_ids, head_dim, rope_max_wavelength, mrope_section
):
    """Compute M-RoPE cos/sin embeddings from position IDs."""
    del mrope_section  # Sections are applied in attention, not here.

    dim = head_dim
    inv_freq = 1.0 / (
        rope_max_wavelength
        ** (ops.cast(ops.arange(0, dim, 2), "float32") / dim)
    )

    position_ids = ops.transpose(
        ops.cast(mrope_position_ids, "float32"), (2, 0, 1)
    )

    inv_freq_expanded = ops.reshape(inv_freq, (1, 1, -1, 1))
    inv_freq_expanded = ops.tile(inv_freq_expanded, (3, 1, 1, 1))

    position_ids_expanded = ops.expand_dims(position_ids, axis=2)

    freqs = ops.matmul(
        ops.cast(inv_freq_expanded, "float32"),
        ops.cast(position_ids_expanded, "float32"),
    )
    freqs = ops.transpose(freqs, (0, 1, 3, 2))

    emb = ops.concatenate([freqs, freqs], axis=-1)

    cos = ops.cos(emb)
    sin = ops.sin(emb)

    return (cos, sin)
