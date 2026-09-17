import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.models.qwen2_vl.qwen2_vl_decoder import (
    Qwen2VLTransformerDecoder,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_layers import (
    Qwen2VLInterleaveEmbeddings,
)


def _qwen2vl_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Qwen2VLBackbone")
class Qwen2VLBackbone(Backbone):
    """Qwen2-VL multimodal backbone.

    Combines a 3D Vision Encoder (ViT with RoPE + PatchMerger) with a
    Qwen2 causal language model decoder. Vision tokens produced by the
    encoder are interleaved into the text embedding sequence at positions
    given by ``vision_indices``, and the decoder applies multimodal RoPE
    (M-RoPE) using per-channel ``position_ids``.

    Args:
        vocabulary_size: int. Vocabulary size of the text model.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads (GQA).
        hidden_dim: int. LLM hidden dimension.
        intermediate_dim: int. Feed-forward intermediate dimension.
        vision_encoder: A ``Qwen2VLVisionEncoder`` instance, or ``None`` for
            a text-only model.
        mrope_section: list of int or None. M-RoPE section sizes
            ``[s_t, s_h, s_w]`` in rotary pairs, summing to ``head_dim // 2``.
            Defaults to ``[16, 24, 24]`` (the value used by all released
            Qwen2-VL checkpoints, whose ``head_dim`` is 128).
        rope_max_wavelength: int. RoPE base wavelength for the text model.
            Defaults to ``1000000``.
        rope_scaling_factor: float. RoPE scaling factor. Defaults to ``1.0``.
        layer_norm_epsilon: float. Epsilon for RMS norm layers. Defaults to
            ``1e-6``.
        dropout: float. Dropout rate. Defaults to ``0``.
        tie_word_embeddings: bool. Whether to tie input/output embeddings.
            Defaults to ``False``.
        use_sliding_window_attention: bool. Whether to use sliding window
            attention. Defaults to ``False``.
        sliding_window_size: int. Sliding window size. Defaults to ``32768``.
        dtype: string or ``keras.mixed_precision.DTypePolicy``.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        vision_encoder=None,
        mrope_section=None,
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0,
        tie_word_embeddings=False,
        use_sliding_window_attention=False,
        sliding_window_size=32768,
        dtype=None,
        **kwargs,
    ):
        if mrope_section is None and vision_encoder is not None:
            # All released Qwen2-VL checkpoints use head_dim=128 and
            # mrope_section=[16, 24, 24] (HF rope_parameters default).
            mrope_section = [16, 24, 24]

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen2vl_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = Qwen2VLTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_qwen2vl_kernel_initializer(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                use_sliding_window_attention=use_sliding_window_attention,
                sliding_window_size=sliding_window_size,
                mrope_section=mrope_section,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = QwenLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )
        self.vision_encoder = vision_encoder
        text_only_model = vision_encoder is None
        if not text_only_model:
            self.interleave_embeddings = Qwen2VLInterleaveEmbeddings(
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="interleave_embeddings",
            )

        # === Functional Model ===
        if not text_only_model:
            # Vision tensors are flat-packed across the batch: the leading
            # dim is the total patch/image count, not the batch dim.
            pixel_values_input = keras.Input(
                shape=(
                    vision_encoder.temporal_patch_size,
                    vision_encoder.patch_size,
                    vision_encoder.patch_size,
                    vision_encoder.in_channels,
                ),
                name="pixel_values",
            )
            image_grid_thw_input = keras.Input(
                shape=(3,), dtype="int32", name="image_grid_thw"
            )
            vision_indices_input = keras.Input(
                shape=(), dtype="int32", name="vision_indices"
            )
            position_ids_input = keras.Input(
                shape=(3, None), dtype="int32", name="position_ids"
            )

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Text embeddings.
        text_embeddings = self.token_embedding(token_id_input)

        # Vision: encoder → interleave into text embeddings.
        if not text_only_model:
            img_embeddings = self.vision_encoder(
                pixel_values_input, image_grid_thw_input
            )
            x = self.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices_input,
            )
        else:
            position_ids_input = None
            x = text_embeddings

        # Transformer layers.
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                decoder_padding_mask=padding_mask_input,
                position_ids=position_ids_input,
            )

        sequence_output = self.layer_norm(x)

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
        if not text_only_model:
            inputs.update(
                {
                    "pixel_values": pixel_values_input,
                    "image_grid_thw": image_grid_thw_input,
                    "vision_indices": vision_indices_input,
                    "position_ids": position_ids_input,
                }
            )

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
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.mrope_section = mrope_section
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.text_only_model = text_only_model

    def _inject_empty_vision_inputs(self, inputs):
        """Inject default empty vision inputs for text-only calls.

        When a multimodal backbone receives text-only inputs (no
        ``pixel_values``, ``image_grid_thw``, ``vision_indices`` or
        ``position_ids`` keys), this injects zero-sized vision tensors and
        sequential position IDs (equivalent to standard 1-D RoPE) so the
        functional graph receives all required keys. This follows the
        Gemma3/Qwen3.5 pattern and allows users to call the backbone with
        only ``token_ids`` and ``padding_mask``.
        """
        if not isinstance(inputs, dict) or self.text_only_model:
            return inputs
        inputs = dict(inputs)  # shallow copy to avoid mutation
        ve = self.vision_encoder
        missing = (
            "pixel_values" not in inputs
            or "image_grid_thw" not in inputs
            or "vision_indices" not in inputs
            or "position_ids" not in inputs
        )
        if missing:
            # Convert all values to tensors first — Keras rejects a
            # nested call argument mixing tensors and non-tensors.
            for key in inputs:
                inputs[key] = ops.convert_to_tensor(inputs[key])
        if "pixel_values" not in inputs:
            inputs["pixel_values"] = ops.zeros(
                (
                    0,
                    ve.temporal_patch_size,
                    ve.patch_size,
                    ve.patch_size,
                    ve.in_channels,
                )
            )
        if "image_grid_thw" not in inputs:
            inputs["image_grid_thw"] = ops.zeros((0, 3), dtype="int32")
        if "vision_indices" not in inputs:
            inputs["vision_indices"] = ops.zeros((0,), dtype="int32")
        if "position_ids" not in inputs:
            batch_size = ops.shape(inputs["token_ids"])[0]
            seq_len = ops.shape(inputs["token_ids"])[1]
            inputs["position_ids"] = ops.broadcast_to(
                ops.cast(ops.reshape(ops.arange(seq_len), (1, 1, -1)), "int32"),
                (batch_size, 3, seq_len),
            )
        return inputs

    def __call__(self, inputs, *args, **kwargs):
        inputs = self._inject_empty_vision_inputs(inputs)
        return super().__call__(inputs, *args, **kwargs)

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
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "vision_encoder": None
                if config["vision_encoder"] is None
                else keras.layers.deserialize(config["vision_encoder"]),
            }
        )
        return super().from_config(config)
