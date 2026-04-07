import keras
from keras import layers
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma4.gemma4_decoder_block import (
    Gemma4TextDecoderBlock,
)
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4InterleaveEmbeddings
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4MeanPooling
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization


@keras_hub_export("keras_hub.models.Gemma4Backbone")
class Gemma4Backbone(Backbone):
    """Gemma4 core network with hyperparameters.

    This backbone implements the Gemma4 model architecture. Gemma4 is a
    multimodal vision-language model (image + text in, text out). The text
    input is encoded with a scaled embedding layer; images are encoded by a
    separate vision transformer (`Gemma4VisionEncoder`). After encoding, image
    embeddings are placed at the correct positions in the text-embedding
    sequence, and the combined sequence is processed by transformer decoder
    layers.

    Compared to Gemma3, Gemma4 introduces:

    * **Four norms per decoder block** — pre + post for both attention and FFW,
      always enabled (no `use_post_*_norm` flags).
    * **Q / K / V normalisation** in attention always on.
    * **Attention scaling = 1.0** — Q/K normalisation provides stability
      instead of the classic `1/sqrt(head_dim)` scaling.
    * **New vision encoder** — uses the same Gemma4 decoder blocks with
      bidirectional attention, 2D learnable position embeddings, and spatial
      average-pooling.
    * **Smaller default sliding window** — 512 tokens (vs. 1 024 in Gemma3).
    * **Audio encoder** — an optional Universal Speech Model (USM) conformer
      that encodes mel spectrograms into audio token embeddings.

    For a higher-level object for text generation see
    `keras_hub.models.Gemma4CausalLM`.

    The default constructor gives a fully customised, randomly initialised
    Gemma4 model. To load preset weights use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        image_size: int. The spatial resolution of images fed to the vision
            encoder (height = width). Must be divisible by
            `patch_size * pool_size` when a `vision_encoder` is provided.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query heads per attention layer.
        num_key_value_heads: int. Number of key/value heads (GQA).
        hidden_dim: int. Hidden state dimension at the end of each layer.
        intermediate_dim: int. First dense layer output dimension in each FFW
            sub-block.
        head_dim: int. Per-head dimension in the decoder attention.
        query_head_dim_normalize: bool. If `True` normalise query pre-attention
            using `head_dim`; otherwise use `hidden_dim / num_query_heads`.
            **Unused in Gemma4 (always Q-normalised via `q_norm`).** Kept for
            API compatibility. Defaults to `True`.
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
        global_head_dim: int or `None`. Per-head dimension used specifically
            for global attention layers. When `None`, `head_dim` is used
            for all layers. Defaults to `None`.
        local_rope_scaling_factor: float. RoPE scaling factor for local layers.
            Defaults to `1.0`.
        global_rope_scaling_factor: float. RoPE scaling factor for global
            layers. Defaults to `1.0`.
        global_rope_partial_rotary_factor: float. Fraction of each head
            dimension that receives rotary position embeddings in global
            attention layers. Only the first
            `int(factor * head_dim)` dimensions are rotated; the remainder are
            left unchanged (NoPE). Local layers always use full RoPE
            (`factor = 1.0`). Defaults to `1.0`.
        vision_encoder: `keras_hub.models.Gemma4VisionEncoder` or `None`. When
            `None` the model processes no images.
        audio_encoder: `keras_hub.models.Gemma4AudioEncoder` or `None`. When
            `None` the model processes no audio.
        num_audio_tokens_per_clip: int or `None`. Number of audio soft tokens
            produced per audio clip (including zero-padded positions). Must be
            provided when `audio_encoder` is not `None`.
        layer_norm_epsilon: float. Epsilon for all RMS norms. Defaults `1e-6`.
        use_bidirectional_attention: bool. When `True` the model uses fully
            bidirectional attention for ALL tokens, e.g. for embedding
            models. This is distinct from `use_vision_bidirectional_attention`
            which only affects vision token attention. Defaults to `False`.
        use_vision_bidirectional_attention: bool. When `True`, vision tokens
            within the same image attend to each other bidirectionally while
            text tokens remain causal. Corresponds to HF
            `use_bidirectional_attention: "vision"` (present in 26B and 31B
            models; `null` for the 2B and 4B models). Defaults to `False`.
        dropout: float. Dropout probability. Defaults to `0`.
        is_embedding_model: bool. When `True` add mean-pooling and dense
            projection heads for embedding models. Defaults to `False`.
        pooling_intermediate_dim: int or `None`. Intermediate dimension of the
            first projection in the pooling head. Required when
            `is_embedding_model=True`.
        embedding_dim: int or `None`. Final embedding dimension. Required when
            `is_embedding_model=True`.
        num_kv_shared_layers: int. Number of trailing decoder layers that
            share K/V projections with the most recent non-shared layer of the
            same attention type. Defaults to `0`.
        num_global_key_value_heads: int or `None`. When set, global attention
            layers use this many K/V heads instead of `num_key_value_heads`
            and enable the K=V projection. Defaults to `None`.
        hidden_size_per_layer_input: int. Size of the per-token, per-layer
            conditioning vector that gates each decoder layer's output.
            Set to `0` to disable. Defaults to `0`.
        vocab_size_per_layer_input: int or `None`. Vocabulary size for the
            per-layer token embedding table. When `None` falls back to
            `vocabulary_size`. Defaults to `None`.
        use_double_wide_mlp: bool. When `True`, KV-shared layers
            (`is_kv_shared_layer=True`) use `2 × intermediate_dim` for their
            FFW sub-block. Defaults to `False`.
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

    Example:
    ```python
    import numpy as np

    # Text-only input.
    model = keras_hub.models.Gemma4Backbone(
        vocabulary_size=262144,
        image_size=768,
        num_layers=26,
        num_query_heads=8,
        num_key_value_heads=4,
        hidden_dim=2304,
        intermediate_dim=9216,
        head_dim=256,
        sliding_window_size=512,
        vision_encoder=None,
        dtype="bfloat16",
    )
    inputs = {
        "token_ids": np.ones((1, 128), dtype="int32"),
        "padding_mask": np.ones((1, 128), dtype="int32"),
    }
    model(inputs)
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
        global_head_dim=None,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        vision_encoder=None,
        audio_encoder=None,
        num_audio_tokens_per_clip=None,
        layer_norm_epsilon=1e-6,
        use_bidirectional_attention=False,
        use_vision_bidirectional_attention=False,
        dropout=0,
        is_embedding_model=False,
        pooling_intermediate_dim=None,
        embedding_dim=None,
        num_kv_shared_layers=0,
        num_global_key_value_heads=None,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=None,
        global_rope_wavelength=None,
        local_rope_wavelength=None,
        global_rope_partial_rotary_factor=1.0,
        use_double_wide_mlp=False,
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

        # Per-layer token-conditioned input.
        # Each decoder layer receives a per-token, per-layer embedding that
        # gates its output.
        if hidden_size_per_layer_input > 0:
            _vocab_per_layer = (
                vocab_size_per_layer_input
                if vocab_size_per_layer_input is not None
                else vocabulary_size
            )
            self.per_layer_token_embedding = keras.layers.Embedding(
                _vocab_per_layer,
                num_layers * hidden_size_per_layer_input,
                dtype=dtype,
                name="per_layer_token_embedding",
            )
            # Projects text embeddings →
            # (num_layers × hidden_size_per_layer_input),
            # scaled by hidden_dim**-0.5.
            self.per_layer_model_projection = keras.layers.Dense(
                num_layers * hidden_size_per_layer_input,
                use_bias=False,
                dtype=dtype,
                name="per_layer_model_projection",
            )
            self.per_layer_projection_norm = RMSNormalization(
                epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="per_layer_projection_norm",
            )

        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        text_only_model = vision_encoder is None and audio_encoder is None
        if vision_encoder is not None:
            self.interleave_embeddings = Gemma4InterleaveEmbeddings(
                num_vision_tokens_per_image=(
                    self.vision_encoder.num_vision_tokens_per_image
                ),
                dtype=dtype,
                name="interleave_embeddings",
            )
        if audio_encoder is not None:
            if num_audio_tokens_per_clip is None:
                raise ValueError(
                    "`num_audio_tokens_per_clip` must be provided when "
                    "`audio_encoder` is not None."
                )
            self.audio_interleave_embeddings = Gemma4InterleaveEmbeddings(
                num_vision_tokens_per_image=num_audio_tokens_per_clip,
                dtype=dtype,
                name="audio_interleave_embeddings",
            )

        # Build transformer layers.
        # Pattern: every 6th layer (index % 6 == 5) is global attention;
        # the rest use (optional) sliding-window local attention.
        # Precompute KV-sharing indices.
        # The last `num_kv_shared_layers` layers reuse K/V from the most
        # recent non-shared layer of the same attention type.
        _first_kv_shared = num_layers - num_kv_shared_layers
        if num_kv_shared_layers > 0:
            _non_shared_types = [
                "global"
                if (j % sliding_window_pattern) == (sliding_window_pattern - 1)
                else "local"
                for j in range(_first_kv_shared)
            ]
            # Map each shared layer index → the absolute index of its KV source.
            _kv_source = {}
            for j in range(_first_kv_shared, num_layers):
                _is_g = (j % sliding_window_pattern) == (
                    sliding_window_pattern - 1
                )
                _type = "global" if _is_g else "local"
                for k in range(len(_non_shared_types) - 1, -1, -1):
                    if _non_shared_types[k] == _type:
                        _kv_source[j] = k
                        break
        else:
            _kv_source = {}

        self.transformer_layers = []
        for i in range(num_layers):
            # A layer is global when it's the last in each group of
            # `sliding_window_pattern` consecutive layers.
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
            is_kv_shared = i in _kv_source
            kv_shared_layer_index = _kv_source.get(i, None)
            # Global attention layers optionally use fewer KV heads + K=V
            # projection when `num_global_key_value_heads` is set.
            use_alt_attn = is_global and num_global_key_value_heads is not None
            # Global layers use proportional (partial) RoPE; local layers get
            # the full RoPE (factor = 1.0).
            layer_rope_partial = (
                global_rope_partial_rotary_factor if is_global else 1.0
            )
            layer_rope_wavelength = rope_wavelength
            layer = Gemma4TextDecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                logit_soft_cap=attention_logit_soft_cap,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
                rope_wavelength=layer_rope_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                rope_partial_rotary_factor=layer_rope_partial,
                use_bidirectional_attention=use_bidirectional_attention,
                use_vision_bidirectional_attention=use_vision_bidirectional_attention,
                is_global_attention=is_global,
                global_head_dim=global_head_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                is_kv_shared_layer=is_kv_shared,
                kv_shared_layer_index=kv_shared_layer_index,
                hidden_size_per_layer_input=hidden_size_per_layer_input,
                attention_k_eq_v=use_alt_attn,
                num_global_key_value_heads=(
                    num_global_key_value_heads if use_alt_attn else None
                ),
                use_double_wide_mlp=use_double_wide_mlp,
                enable_moe_block=enable_moe_block,
                num_experts=num_experts,
                expert_intermediate_dim=expert_intermediate_dim,
                num_experts_per_token=num_experts_per_token,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)

        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        # === Functional Model ===

        # Vision inputs.
        if vision_encoder is not None:
            pixel_values_input = keras.Input(
                shape=(None, None, None),
                name="pixel_values",
            )
            pixel_position_ids_input = keras.Input(
                shape=(None, None, 2), dtype="int32", name="pixel_position_ids"
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            vision_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_mask"
            )

        # Audio inputs.
        if audio_encoder is not None:
            audio_mel_input = keras.Input(
                shape=(None, None, audio_encoder.input_feat_size),
                name="audio_mel",
            )
            audio_mel_mask_input = keras.Input(
                shape=(None, None), dtype="int32", name="audio_mel_mask"
            )
            audio_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="audio_indices"
            )

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Text embeddings.
        text_embeddings = self.token_embedding(token_id_input)

        # Interleave image embeddings. Pre-scale by 1/sqrt(hidden_dim) so that
        # after the global x *= sqrt(hidden_dim) below, vision positions remain
        # at their natural (unscaled) embed_vision magnitude.
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

        # Interleave audio embeddings (same pre-scaling as vision).
        if audio_encoder is not None:
            audio_embeddings = self.audio_encoder(
                audio_mel_input,
                ops.cast(audio_mel_mask_input, "bool"),
            )
            audio_embeddings = audio_embeddings * ops.cast(
                float(hidden_dim) ** -0.5, audio_embeddings.dtype
            )
            x = self.audio_interleave_embeddings(
                image_embeddings=audio_embeddings,
                text_embeddings=x,
                vision_indices=audio_indices_input,
            )

        # Per-layer token embeddings. Vision positions use pad_token_id (0),
        # mirroring HF's llm_input_ids masking before embed_tokens_per_layer.
        if hidden_size_per_layer_input > 0:
            _hpl = hidden_size_per_layer_input
            _per_layer_ids = token_id_input
            if vision_encoder is not None:
                _per_layer_ids = ops.where(
                    ops.cast(vision_mask_input, "bool"),
                    ops.zeros_like(_per_layer_ids),
                    _per_layer_ids,
                )
            _per_emb = self.per_layer_token_embedding(_per_layer_ids)
            _per_emb = ops.cast(_per_emb, x.dtype)
            _per_emb = _per_emb * ops.cast(float(_hpl) ** 0.5, _per_emb.dtype)
            per_layer_emb_flat = _per_emb
        else:
            per_layer_emb_flat = None

        # Global scale: text positions → token_embedding * sqrt(hidden_dim);
        # vision/audio positions remain at their pre-scaled embed magnitude.
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)

        # Per-layer model projection, computed after the global scale.
        if hidden_size_per_layer_input > 0:
            _per_proj = self.per_layer_model_projection(x)
            _per_proj = _per_proj * ops.cast(
                float(hidden_dim) ** -0.5, _per_proj.dtype
            )
            per_layer_proj_flat = _per_proj
        else:
            per_layer_proj_flat = None

        # Decoder layers.
        _hpl = hidden_size_per_layer_input
        shared_kv_tensors = {}
        for i, transformer_layer in enumerate(self.transformer_layers):
            if per_layer_proj_flat is not None:
                # Slice the i-th (hpl,) block: apply projection norm to proj
                # alone, then combine with emb and scale by 2^-0.5.
                # This matches call_with_cache and the HF Gemma4 formula:
                #   per_layer_input = (norm(proj) + emb) * 2^-0.5
                proj_i = per_layer_proj_flat[:, :, i * _hpl : (i + 1) * _hpl]
                emb_i = per_layer_emb_flat[:, :, i * _hpl : (i + 1) * _hpl]
                proj_i_normed = self.per_layer_projection_norm(proj_i)
                per_layer_input_i = (proj_i_normed + emb_i) * ops.cast(
                    2.0**-0.5, proj_i.dtype
                )
            else:
                per_layer_input_i = None

            shared_kv = None
            if getattr(transformer_layer, "is_kv_shared_layer", False):
                idx = getattr(transformer_layer, "kv_shared_layer_index", None)
                if idx is not None:
                    shared_kv = shared_kv_tensors.get(idx, None)

            x, new_cache = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                vision_mask=(
                    None if vision_encoder is None else vision_mask_input
                ),
                per_layer_input=per_layer_input_i,
                shared_kv=shared_kv,
            )
            shared_kv_tensors[i] = new_cache
        sequence_output = self.layer_norm(x)

        if is_embedding_model:
            if embedding_dim is None or pooling_intermediate_dim is None:
                raise ValueError(
                    "Must specify embedding_dim and pooling_intermediate_dim."
                )

            pooled_output = Gemma4MeanPooling(dtype=dtype, name="mean_pooling")(
                sequence_output, padding_mask=padding_mask_input
            )

            pooled_output = layers.Dense(
                pooling_intermediate_dim,
                dtype=dtype,
                name="pooling_dense_1",
                use_bias=False,
            )(pooled_output)

            pooled_output = layers.Dense(
                embedding_dim,
                dtype=dtype,
                name="embedding_projection",
                use_bias=False,
            )(pooled_output)

            pooled_output = layers.UnitNormalization(
                axis=-1, dtype=dtype, name="unit_normalization"
            )(pooled_output)

            outputs = {
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            }
        else:
            outputs = sequence_output

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
        if vision_encoder is not None:
            inputs.update(
                {
                    "pixel_values": pixel_values_input,
                    "pixel_position_ids": pixel_position_ids_input,
                    "vision_indices": vision_indices_input,
                    "vision_mask": vision_mask_input,
                }
            )
        if audio_encoder is not None:
            inputs.update(
                {
                    "audio_mel": audio_mel_input,
                    "audio_mel_mask": audio_mel_mask_input,
                    "audio_indices": audio_indices_input,
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
        self.is_embedding_model = is_embedding_model
        self.pooling_intermediate_dim = pooling_intermediate_dim
        self.embedding_dim = embedding_dim
        self.num_audio_tokens_per_clip = num_audio_tokens_per_clip
        self.num_kv_shared_layers = num_kv_shared_layers
        self.num_global_key_value_heads = num_global_key_value_heads
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.global_rope_wavelength = global_rope_wavelength
        self.local_rope_wavelength = local_rope_wavelength
        self.global_rope_partial_rotary_factor = (
            global_rope_partial_rotary_factor
        )
        self.use_double_wide_mlp = use_double_wide_mlp
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.expert_intermediate_dim = expert_intermediate_dim
        self.num_experts_per_token = num_experts_per_token

        # Keep `num_vision_tokens_per_image` and `text_only_model` accessible.
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
                "global_head_dim": self.global_head_dim,
                "local_rope_scaling_factor": self.local_rope_scaling_factor,
                "global_rope_scaling_factor": self.global_rope_scaling_factor,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
                "audio_encoder": None
                if self.audio_encoder is None
                else keras.layers.serialize(self.audio_encoder),
                "num_audio_tokens_per_clip": self.num_audio_tokens_per_clip,
                "use_bidirectional_attention": self.use_bidirectional_attention,
                "use_vision_bidirectional_attention": (
                    self.use_vision_bidirectional_attention
                ),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "is_embedding_model": self.is_embedding_model,
                "pooling_intermediate_dim": self.pooling_intermediate_dim,
                "embedding_dim": self.embedding_dim,
                "num_kv_shared_layers": self.num_kv_shared_layers,
                "num_global_key_value_heads": self.num_global_key_value_heads,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
                "global_rope_wavelength": self.global_rope_wavelength,
                "local_rope_wavelength": self.local_rope_wavelength,
                "global_rope_partial_rotary_factor": (
                    self.global_rope_partial_rotary_factor
                ),
                "use_double_wide_mlp": self.use_double_wide_mlp,
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
                "audio_encoder": None
                if config.get("audio_encoder") is None
                else keras.layers.deserialize(config["audio_encoder"]),
            }
        )
        return super().from_config(config)
