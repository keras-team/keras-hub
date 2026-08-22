import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_self_conditioning import (  # noqa: E501
    DiffusionGemmaSelfConditioning,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_transformer_layer import (  # noqa: E501
    DiffusionGemmaTransformerLayer,
)
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4InterleaveEmbeddings
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization


@keras_hub_export("keras_hub.models.DiffusionGemmaBackbone")
class DiffusionGemmaBackbone(Backbone):
    """DiffusionGemma core network with hyperparameters.

    This backbone implements the DiffusionGemma model architecture.
    DiffusionGemma extends Gemma4 with discrete block-diffusion generation
    support: the backbone is called twice per denoising iteration — once as a
    causal encoder to encode the prompt KV caches, and then repeatedly as a
    bidirectional decoder over a fixed-length canvas of tokens.

    Two diffusion-specific features are always enabled:

    * **`encoder_layer_scalar`** — each decoder block has a second
      non-trainable scalar used during the causal encoder pass.  The standard
      `layer_scalar` is used for decoder passes.
    * **`DiffusionGemmaSelfConditioning`** — refines canvas embeddings at the
      start of each denoising step using the logits predicted in the previous
      step.

    In all other respects the architecture matches `Gemma4Backbone`.  Shared
    Gemma4 layer classes (norms, vision encoder, etc.) are
    imported directly from `keras_hub.models.gemma4`.

    The default constructor gives a fully customised, randomly initialised
    DiffusionGemmaBackbone.  To load preset weights use `from_preset`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        image_size: int. The spatial resolution of images (height = width).
            Stored as a config value for serialization purposes only; it does
            not affect the backbone's forward pass. Image patching and resizing
            are handled by `keras_hub.layers.Gemma4ImageConverter` before data
            reaches the backbone. The `vision_encoder` has its own `image_size`
            parameter that controls position embedding sizes.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query heads per attention layer.
        num_key_value_heads: int. Number of key/value heads (GQA).
        hidden_dim: int. Hidden state dimension at the end of each layer.
        intermediate_dim: int. First dense layer output dimension in each FFW
            sub-block.
        head_dim: int. Per-head dimension in the decoder attention.
        query_head_dim_normalize: bool. If `True` normalise query pre-attention
            using `head_dim`; otherwise use `hidden_dim / num_query_heads`.
            **Unused in DiffusionGemma (always Q-normalised via `q_norm`).**
            Kept for API compatibility. Defaults to `True`.
        attention_logit_soft_cap: `None` or float. Tanh soft-cap on attention
            logits. Defaults to `None`.
        final_logit_soft_cap: `None` or float. Tanh soft-cap on output logits.
            Defaults to `None`.
        use_sliding_window_attention: bool. Whether to use sliding-window
            attention on the local layers. Defaults to `True`.
        sliding_window_size: int. Size of the local attention window. Defaults
            to `512`.
        sliding_window_pattern: int. Repeat period of the local/global
            attention pattern. The last layer in each group of this many
            consecutive layers uses global attention; all others use local
            (sliding-window) attention. Defaults to `6`.
        layer_types: list of str or `None`. Explicit specification of the
            attention type for every layer sequentially
            (e.g. `"full_attention"`, `"sliding_attention"`). When `None`,
            type sequence is derived from `sliding_window_pattern`.
            Defaults to `None`.
        global_head_dim: int or `None`. Per-head dimension used specifically
            for global attention layers. When `None`, `head_dim` is used
            for all layers. Defaults to `None`.
        local_rope_scaling_factor: float. RoPE scaling factor for local layers.
            Defaults to `1.0`.
        global_rope_scaling_factor: float. RoPE scaling factor for global
            layers. Defaults to `1.0`.
        vision_encoder: `keras_hub.models.Gemma4VisionEncoder` or `None`. When
            `None` the model processes no images.
        layer_norm_epsilon: float. Epsilon for all RMS norms. Defaults to
            `1e-6`.
        use_bidirectional_attention: bool. When `True` the model uses fully
            bidirectional attention for ALL tokens. Defaults to `False`.
        use_vision_bidirectional_attention: bool. When `True`, vision tokens
            within the same image attend to each other bidirectionally while
            text tokens remain causal. Defaults to `False`.
        dropout: float. Dropout probability. Defaults to `0`.
        num_global_key_value_heads: int or `None`. When set, global attention
            layers use this many K/V heads instead of `num_key_value_heads`
            and enable the K=V projection. Defaults to `None`.
        global_rope_wavelength: float or `None`. Base RoPE wavelength for
            global attention layers. When `None`, defaults to `1_000_000.0`.
            Defaults to `None`.
        local_rope_wavelength: float or `None`. Base RoPE wavelength for
            local (sliding-window) attention layers. When `None`, defaults to
            `10_000.0`. Defaults to `None`.
        global_rope_partial_rotary_factor: float. Fraction of each head
            dimension that receives rotary position embeddings in global
            attention layers. Only the first
            `int(factor * head_dim)` dimensions are rotated; the remainder are
            left unchanged (NoPE). Local layers always use full RoPE
            (`factor = 1.0`). Defaults to `1.0`.
        enable_moe_block: bool. When `True`, every decoder layer runs a
            parallel Mixture-of-Experts path alongside the dense FFW path.
            The two outputs are summed before the shared post-FFW norm.
            Requires `num_experts` and `expert_intermediate_dim` to be set.
            Defaults to `False`.
        num_experts: int or `None`. Total number of expert MLPs in the MoE
            bank. Required when `enable_moe_block=True`. Defaults to `None`.
        expert_intermediate_dim: int or `None`. Intermediate dimension of each
            expert MLP. Required when `enable_moe_block=True`.
            Defaults to `None`.
        num_experts_per_token: int. Top-k experts selected per token by the
            MoE router. Defaults to `8`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. Compute dtype.
            Defaults to `None`.

    Examples:

    ```python
    model = keras_hub.models.DiffusionGemmaBackbone.from_preset(
        "diffusion_gemma_26b_a4b_it"
    )
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        image_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        query_head_dim_normalize=True,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=True,
        sliding_window_size=512,
        sliding_window_pattern=6,
        layer_types=None,
        global_head_dim=None,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        vision_encoder=None,
        layer_norm_epsilon=1e-6,
        use_bidirectional_attention=False,
        use_vision_bidirectional_attention=False,
        dropout=0,
        num_global_key_value_heads=None,
        global_rope_wavelength=None,
        local_rope_wavelength=None,
        global_rope_partial_rotary_factor=1.0,
        enable_moe_block=False,
        num_experts=None,
        expert_intermediate_dim=None,
        num_experts_per_token=8,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
            ),
            dtype=dtype,
            logit_soft_cap=final_logit_soft_cap,
            name="token_embedding",
        )

        self.vision_encoder = vision_encoder
        self.layer_types = layer_types
        text_only_model = vision_encoder is None
        if vision_encoder is not None:
            self.interleave_embeddings = Gemma4InterleaveEmbeddings(
                num_vision_tokens_per_image=(
                    self.vision_encoder.num_vision_tokens_per_image
                ),
                dtype=dtype,
                name="interleave_embeddings",
            )

        # Build DiffusionGemmaTransformerLayer layers
        self.transformer_layers = []
        for i in range(num_layers):
            if layer_types is not None:
                is_global = layer_types[i] == "full_attention"
            else:
                is_global = (i % sliding_window_pattern) == (
                    sliding_window_pattern - 1
                )
            sliding_window = use_sliding_window_attention and not is_global
            rope_wavelength = (
                (global_rope_wavelength or 1_000_000.0)
                if is_global
                else (local_rope_wavelength or 10_000.0)
            )
            rope_scaling_factor = (
                global_rope_scaling_factor
                if is_global
                else local_rope_scaling_factor
            )
            use_alt_attn = is_global and num_global_key_value_heads is not None
            layer_rope_partial = (
                global_rope_partial_rotary_factor if is_global else 1.0
            )
            layer = DiffusionGemmaTransformerLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                logit_soft_cap=attention_logit_soft_cap,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
                rope_wavelength=rope_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                rope_partial_rotary_factor=layer_rope_partial,
                use_bidirectional_attention=use_bidirectional_attention,
                use_vision_bidirectional_attention=use_vision_bidirectional_attention,
                is_global_attention=is_global,
                global_head_dim=global_head_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                attention_k_eq_v=use_alt_attn,
                num_global_key_value_heads=(
                    num_global_key_value_heads if use_alt_attn else None
                ),
                enable_moe_block=enable_moe_block,
                num_experts=num_experts,
                expert_intermediate_dim=expert_intermediate_dim,
                num_experts_per_token=num_experts_per_token,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)

        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if (i % sliding_window_pattern) == (sliding_window_pattern - 1)
                else "sliding_attention"
                for i in range(num_layers)
            ]

        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        # Self-conditioning layer
        self.diffusion_self_conditioning = DiffusionGemmaSelfConditioning(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="diffusion_self_conditioning",
        )
        # Inject a reference to the shared token_embedding without Keras layer
        # tracking. Normal setattr would register it as a sub-layer of
        # diffusion_self_conditioning, causing duplicate weight serialization.
        # from_config() re-injects this reference after deserialization.
        object.__setattr__(
            self.diffusion_self_conditioning,
            "_token_embedding_layer",
            self.token_embedding,
        )
        self.diffusion_self_conditioning.build((None, None, hidden_dim))

        # === Functional Model ===

        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        if vision_encoder is not None:
            pixel_position_ids_input = keras.Input(
                shape=(None, None, 2), dtype="int32", name="pixel_position_ids"
            )
            pixel_values_input = keras.Input(
                shape=(None, None, None),
                name="pixel_values",
            )

        position_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="position_ids"
        )
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )

        if vision_encoder is not None:
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            vision_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_mask"
            )

        # Text embeddings.
        text_embeddings = self.token_embedding(token_id_input)

        if vision_encoder is not None:
            img_embeddings = self.vision_encoder(
                {
                    "pixel_values": pixel_values_input,
                    "pixel_position_ids": pixel_position_ids_input,
                }
            )
            img_embeddings = img_embeddings * ops.cast(
                float(hidden_dim) ** -0.5, img_embeddings.dtype
            )
            x = self.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices_input,
            )
        else:
            x = text_embeddings

        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)

        # Decoder layers.
        for transformer_layer in self.transformer_layers:
            x, _ = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                vision_mask=(
                    None if vision_encoder is None else vision_mask_input
                ),
                positions=position_ids_input,
            )

        # Wire diffusion_self_conditioning into the functional graph so Keras
        # tracks its weights. The zero-multiply makes this a no-op at runtime;
        # the layer is called with real inputs in _prepare_canvas_embeds()
        # during generation.
        _zero_prev = ops.tile(
            ops.zeros_like(x[:, :1, :1]), [1, 1, vocabulary_size]
        )
        _sc_out = self.diffusion_self_conditioning(x[:, :1], _zero_prev)
        x = x + ops.zeros_like(x[:, :1]) * _sc_out

        sequence_output = self.layer_norm(x)
        outputs = sequence_output

        inputs = {
            "padding_mask": padding_mask_input,
            "position_ids": position_ids_input,
            "token_ids": token_id_input,
        }

        if vision_encoder is not None:
            inputs.update(
                {
                    "pixel_position_ids": pixel_position_ids_input,
                    "pixel_values": pixel_values_input,
                    "vision_indices": vision_indices_input,
                    "vision_mask": vision_mask_input,
                }
            )
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.query_head_dim_normalize = query_head_dim_normalize
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.final_logit_soft_cap = final_logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.sliding_window_pattern = sliding_window_pattern
        self.global_head_dim = global_head_dim
        self.local_rope_scaling_factor = local_rope_scaling_factor
        self.global_rope_scaling_factor = global_rope_scaling_factor
        self.use_bidirectional_attention = use_bidirectional_attention
        self.use_vision_bidirectional_attention = (
            use_vision_bidirectional_attention
        )
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.num_global_key_value_heads = num_global_key_value_heads
        self.global_rope_wavelength = global_rope_wavelength
        self.local_rope_wavelength = local_rope_wavelength
        self.global_rope_partial_rotary_factor = (
            global_rope_partial_rotary_factor
        )
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.expert_intermediate_dim = expert_intermediate_dim
        self.num_experts_per_token = num_experts_per_token

        if vision_encoder is not None:
            self.num_vision_tokens_per_image = (
                self.vision_encoder.num_vision_tokens_per_image
            )
        self.text_only_model = text_only_model

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "query_head_dim_normalize": self.query_head_dim_normalize,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "final_logit_soft_cap": self.final_logit_soft_cap,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "sliding_window_pattern": self.sliding_window_pattern,
                "layer_types": self.layer_types,
                "global_head_dim": self.global_head_dim,
                "local_rope_scaling_factor": self.local_rope_scaling_factor,
                "global_rope_scaling_factor": self.global_rope_scaling_factor,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
                "use_bidirectional_attention": self.use_bidirectional_attention,
                "use_vision_bidirectional_attention": (
                    self.use_vision_bidirectional_attention
                ),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "num_global_key_value_heads": self.num_global_key_value_heads,
                "global_rope_wavelength": self.global_rope_wavelength,
                "local_rope_wavelength": self.local_rope_wavelength,
                "global_rope_partial_rotary_factor": (
                    self.global_rope_partial_rotary_factor
                ),
                "enable_moe_block": self.enable_moe_block,
                "num_experts": self.num_experts,
                "expert_intermediate_dim": self.expert_intermediate_dim,
                "num_experts_per_token": self.num_experts_per_token,
            }
        )
        return config

    def default_lora_layer_names(self):
        target_names = super().default_lora_layer_names()
        if not self.text_only_model:
            target_names += ["query_proj", "value_proj"]
        return target_names

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "vision_encoder": None
                if config["vision_encoder"] is None
                else keras.layers.deserialize(config["vision_encoder"]),
            }
        )
        model = super().from_config(config)
        object.__setattr__(
            model.diffusion_self_conditioning,
            "_token_embedding_layer",
            model.token_embedding,
        )
        return model
