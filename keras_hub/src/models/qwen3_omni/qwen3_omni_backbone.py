import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_moe.qwen3_moe_layernorm import Qwen3MoeLayerNorm
from keras_hub.src.models.qwen3_omni.qwen3_omni_decoder import (
    Qwen3OmniTransformerDecoder,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_layers import (
    Qwen3OmniInterleaveEmbeddings,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_layers import (
    Qwen3OmniVisualPosMask,
)


def _expand_position_ids_from_padding_mask(padding_mask):
    """Build text-only ``(3, batch, seq_len)`` M-RoPE position ids.

    Padding positions clamp to ``1`` so they never read out-of-range
    rotary frequencies; the same scalar broadcasts to all 3 channels.
    """
    mask = ops.cast(padding_mask, "int32")
    positions = ops.cumsum(mask, axis=-1) - 1
    positions = ops.where(mask == 0, ops.ones_like(positions), positions)
    return ops.stack([positions] * 3, axis=0)


def _vision_indices_to_mask(vision_indices, reference_tensor):
    """Convert flat vision indices into a ``(batch, seq_len)`` bool mask.

    The buffer is shaped from ``reference_tensor`` to avoid Python-level
    multiplication of dynamic shape values. Empty indices yield an
    all-False mask (no-op DeepStack injection).
    """
    flat = ops.reshape(
        ops.cast(ops.zeros_like(reference_tensor[..., 0]), "int32"), (-1,)
    )
    indices = ops.reshape(ops.cast(vision_indices, "int32"), (-1,))
    flat = ops.scatter_update(
        flat, ops.expand_dims(indices, axis=-1), ops.ones_like(indices)
    )
    return ops.cast(
        ops.reshape(
            flat,
            (ops.shape(reference_tensor)[0], ops.shape(reference_tensor)[1]),
        ),
        "bool",
    )


@keras_hub_export("keras_hub.models.Qwen3OmniBackbone")
class Qwen3OmniBackbone(Backbone):
    """Qwen3-Omni multimodal Transformer backbone.

    Implements the Qwen3-Omni Thinker text decoder with optional vision
    and audio encoders. Vision / audio embeddings are scattered into the
    text sequence at flat ``vision_indices`` / ``audio_indices``
    positions; DeepStack vision features are additively injected into
    the early decoder layers at the same visual positions.

    Args:
        vocabulary_size: int. Token vocabulary size.
        num_layers: int. Number of decoder layers.
        num_query_heads: int. Number of attention query heads.
        num_key_value_heads: int. Number of key / value heads.
        head_dim: int. Per-head attention dimension.
        hidden_dim: int. Decoder hidden dimension.
        intermediate_dim: int. Dense FFN intermediate size.
        moe_intermediate_dim: int. Per-expert intermediate size.
        num_experts: int. Experts per MoE block.
        num_experts_per_tok: int. Top-k experts per token.
        mrope_section: 3-tuple. M-RoPE ``(t, h, w)`` split sizes; must
            sum to ``head_dim // 2``.
        rope_max_wavelength, rope_scaling_factor, rope_attention_scaling:
            RoPE / scaling parameters.
        layer_norm_epsilon: float. Epsilon for RMS norms.
        dropout: float. Attention dropout.
        tie_word_embeddings: bool. Tie LM head to input embedding.
        norm_topk_prob: bool. Renormalise top-k router probabilities.
        decoder_sparse_step: int. Layer cadence for sparse MoE blocks.
        sliding_window_size: int or None. Sliding-window attention; ``None``
            for current Thinker presets.
        router_aux_loss_coefficient: float. MoE load-balancing coefficient.
        mlp_only_layers: list[int] or None. Layers that use a dense MLP
            instead of a sparse MoE block.
        position_id_per_seconds: int. Audio / video timestamp -> rotary
            position scale factor (used by the preprocessor / causal LM).
        audio_encoder, vision_encoder: optional encoders for the
            respective modalities.
        dtype: str or ``keras.mixed_precision.DTypePolicy``.
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
        moe_intermediate_dim,
        num_experts,
        num_experts_per_tok,
        mrope_section,
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        rope_attention_scaling=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        tie_word_embeddings=False,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        sliding_window_size=None,
        router_aux_loss_coefficient=0.001,
        mlp_only_layers=None,
        position_id_per_seconds=25,
        audio_encoder=None,
        vision_encoder=None,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )

        if not mlp_only_layers:
            mlp_only_layers = []

        self.transformer_layers = []
        for i in range(num_layers):
            is_sparse_mlp = (
                (i not in mlp_only_layers)
                and num_experts > 0
                and (i + 1) % decoder_sparse_step == 0
            )
            layer = Qwen3OmniTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                moe_intermediate_dim=moe_intermediate_dim,
                head_dim=head_dim,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
                norm_top_k_prob=norm_topk_prob,
                mrope_section=mrope_section,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                rope_attention_scaling=rope_attention_scaling,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                sliding_window_size=sliding_window_size,
                router_aux_loss_coefficient=router_aux_loss_coefficient,
                is_sparse_mlp=is_sparse_mlp,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = Qwen3MoeLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder
        self.has_vision = vision_encoder is not None
        self.has_audio = audio_encoder is not None
        self.is_multimodal = self.has_vision or self.has_audio

        if self.is_multimodal:
            self.interleave_embeddings = Qwen3OmniInterleaveEmbeddings(
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="interleave_embeddings",
            )
            self.visual_pos_mask = Qwen3OmniVisualPosMask(
                dtype=dtype,
                name="visual_pos_mask",
            )

        # === Functional graph ===
        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        graph_inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        text_embeddings = self.token_embedding(token_ids_input)

        vision_embeddings = None
        vision_indices_input = None
        deepstack_features = None
        if self.has_vision:
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
            graph_inputs["pixel_values"] = pixel_values_input
            graph_inputs["image_grid_thw"] = image_grid_thw_input
            graph_inputs["vision_indices"] = vision_indices_input

            vision_outputs = self.vision_encoder(
                {
                    "pixel_values": pixel_values_input,
                    "grid_thw": image_grid_thw_input,
                }
            )
            vision_embeddings = vision_outputs["pooler_output"]
            deepstack_features = vision_outputs.get("deepstack_features", None)

        audio_embeddings = None
        audio_indices_input = None
        if self.has_audio:
            audio_features_input = keras.Input(
                shape=(None, audio_encoder.num_mel_bins),
                name="audio_features",
            )
            audio_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="audio_indices"
            )
            graph_inputs["audio_features"] = audio_features_input
            graph_inputs["audio_indices"] = audio_indices_input

            audio_embeddings = self.audio_encoder(
                {"input_features": audio_features_input}
            )

        if self.is_multimodal:
            x = self.interleave_embeddings(
                text_embeddings=text_embeddings,
                vision_embeddings=vision_embeddings,
                vision_indices=vision_indices_input,
                audio_embeddings=audio_embeddings,
                audio_indices=audio_indices_input,
            )
        else:
            x = text_embeddings

        # Text-only positions derived from padding mask; multimodal
        # M-RoPE positions are threaded in by the causal LM.
        position_ids = _expand_position_ids_from_padding_mask(
            padding_mask_input
        )

        visual_pos_mask_tensor = None
        if self.has_vision and deepstack_features is not None:
            visual_pos_mask_tensor = self.visual_pos_mask(
                vision_indices_input, text_embeddings
            )

        for i, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(
                x,
                position_ids=position_ids,
                decoder_padding_mask=padding_mask_input,
            )
            if (
                visual_pos_mask_tensor is not None
                and deepstack_features is not None
                and i < len(deepstack_features)
            ):
                x = self._deepstack_process(
                    x, visual_pos_mask_tensor, deepstack_features[i]
                )
        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs=graph_inputs,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.moe_intermediate_dim = moe_intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.router_aux_loss_coefficient = router_aux_loss_coefficient
        self.mlp_only_layers = mlp_only_layers or []
        self.mrope_section = mrope_section
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_attention_scaling = rope_attention_scaling
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.sliding_window_size = sliding_window_size
        self.position_id_per_seconds = position_id_per_seconds

    def _fill_missing_multimodal_inputs(self, inputs):
        """Lets a text-only forward work through a multimodal backbone."""
        if not (isinstance(inputs, dict) and self.is_multimodal):
            return inputs
        # Normalise every value to a tensor before injecting defaults to
        # keep Keras's type-uniformity check happy.
        inputs = {k: ops.convert_to_tensor(v) for k, v in inputs.items()}
        batch_size = inputs["token_ids"].shape[0]
        if self.has_vision:
            ve = self.vision_encoder
            inputs.setdefault(
                "pixel_values",
                ops.zeros(
                    (
                        batch_size,
                        0,
                        ve.temporal_patch_size,
                        ve.patch_size,
                        ve.patch_size,
                        ve.in_channels,
                    )
                ),
            )
            inputs.setdefault(
                "image_grid_thw",
                ops.zeros((batch_size, 0, 3), dtype="int32"),
            )
            inputs.setdefault(
                "vision_indices",
                ops.zeros((batch_size, 0), dtype="int32"),
            )
        if self.has_audio:
            ae = self.audio_encoder
            inputs.setdefault(
                "audio_features",
                ops.zeros((batch_size, 0, ae.num_mel_bins)),
            )
            inputs.setdefault(
                "audio_indices",
                ops.zeros((batch_size, 0), dtype="int32"),
            )
        return inputs

    def __call__(self, inputs, *args, **kwargs):
        inputs = self._fill_missing_multimodal_inputs(inputs)
        return super().__call__(inputs, *args, **kwargs)

    def _standardize_inputs(self, inputs):
        inputs = self._fill_missing_multimodal_inputs(inputs)
        return super()._standardize_inputs(inputs)

    def call(self, inputs, *args, **kwargs):
        inputs = self._fill_missing_multimodal_inputs(inputs)
        return super().call(inputs, *args, **kwargs)

    def stateless_call(
        self,
        trainable_variables,
        non_trainable_variables,
        *args,
        return_losses=False,
        **kwargs,
    ):
        if args:
            args = (
                self._fill_missing_multimodal_inputs(args[0]),
                *args[1:],
            )
        return super().stateless_call(
            trainable_variables,
            non_trainable_variables,
            *args,
            return_losses=return_losses,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "head_dim": self.head_dim,
                "intermediate_dim": self.intermediate_dim,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_attention_scaling": self.rope_attention_scaling,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "norm_topk_prob": self.norm_topk_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "sliding_window_size": self.sliding_window_size,
                "router_aux_loss_coefficient": (
                    self.router_aux_loss_coefficient
                ),
                "mlp_only_layers": self.mlp_only_layers,
                "position_id_per_seconds": self.position_id_per_seconds,
                "audio_encoder": (
                    keras.saving.serialize_keras_object(self.audio_encoder)
                    if self.audio_encoder is not None
                    else None
                ),
                "vision_encoder": (
                    keras.saving.serialize_keras_object(self.vision_encoder)
                    if self.vision_encoder is not None
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("audio_encoder") is not None and isinstance(
            config["audio_encoder"], dict
        ):
            config["audio_encoder"] = keras.layers.deserialize(
                config["audio_encoder"]
            )
        if config.get("vision_encoder") is not None and isinstance(
            config["vision_encoder"], dict
        ):
            config["vision_encoder"] = keras.layers.deserialize(
                config["vision_encoder"]
            )
        return super().from_config(config)

    def _deepstack_process(
        self, hidden_states, visual_pos_masks, visual_embeds
    ):
        """Add DeepStack vision features to decoder hidden states.

        ``visual_embeds`` is padded with a trailing zero row so
        ``take_along_axis`` stays valid when no visual tokens are
        present (text-only forward through a multimodal backbone).
        """
        mask_int = ops.cast(visual_pos_masks, "int32")
        tokens_per_item = ops.sum(mask_int, axis=1)
        batch_offset = ops.cumsum(tokens_per_item, axis=0) - tokens_per_item
        source_indices = ops.maximum(
            ops.cumsum(mask_int, axis=1)
            + ops.expand_dims(batch_offset, axis=1)
            - 1,
            0,
        )
        visual_values = ops.take_along_axis(
            ops.pad(visual_embeds, [[0, 0], [0, 1], [0, 0]]),
            ops.expand_dims(source_indices, -1),
            axis=1,
        )
        mask_expanded = ops.cast(
            ops.expand_dims(visual_pos_masks, -1), hidden_states.dtype
        )
        return hidden_states + (
            ops.cast(visual_values, hidden_states.dtype) * mask_expanded
        )
