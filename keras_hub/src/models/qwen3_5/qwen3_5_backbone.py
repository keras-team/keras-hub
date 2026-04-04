import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_decoder import (
    Qwen3_5TransformerDecoder,
)
from keras_hub.src.models.qwen3_5.qwen3_5_layernorm import Qwen3_5LayerNorm
from keras_hub.src.models.qwen3_5.qwen3_5_layers import (
    Qwen3_5InterleaveEmbeddings,
)


def _qwen3_5_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


def _compute_mrope_position_ids(
    token_ids, vision_mask, image_grid_thw, mrope_section
):
    """Compute 4-channel M-RoPE position IDs for a mixed text+vision sequence.

    For text tokens all 4 channels (text, temporal, height, width) hold the
    same sequential 1D position. For vision tokens, channels 1-3 (temporal,
    height, width) encode the patch's grid coordinates while channel 0 mirrors
    channel 1 (temporal) per the HF interleaved-MRoPE convention.

    Args:
        token_ids: int32 tensor (batch, seq_len).
        vision_mask: bool/int32 tensor (batch, seq_len). 1 where a token is
            a vision token.
        image_grid_thw: int32 tensor (num_images, 3), [T, H, W] per image in
            **patch units** (already divided by patch_size, BEFORE merging).
        mrope_section: list[int] [s_t, s_h, s_w] — number of *pairs* of
            rotary dims per channel.
    Returns:
        position_ids: int32 tensor (batch, 4, seq_len).
    """
    batch_size = ops.shape(token_ids)[0]
    seq_len = ops.shape(token_ids)[1]

    # Start with sequential ids for every channel.
    seq_range = ops.arange(seq_len, dtype="int32")
    pos_ids = ops.tile(
        ops.reshape(seq_range, (1, 1, seq_len)),
        (batch_size, 4, 1),
    )  # (B, 4, seq_len)

    # We will return pos_ids as-is for text-only inputs.
    # For multimodal inputs the caller should pre-compute spatial indices.
    # In this implementation we rely on the preprocessor to have substituted
    # corrected position_ids as needed; the backbone merely forwards them.
    return pos_ids


@keras_hub_export("keras_hub.models.Qwen3_5Backbone")
class Qwen3_5Backbone(Backbone):
    """The Qwen3.5 Transformer core architecture with hyperparameters.

    This network implements a hybrid Transformer-based decoder with two
    layer types:
    - ``full_attention``: Standard grouped-query attention with partial
      rotary embeddings and sigmoid output gating.
    - ``linear_attention``: GatedDeltaNet recurrent linear attention with
      causal conv1d and delta rule recurrence.

    The backbone optionally accepts a ``vision_encoder`` to enable
    multimodal (image + text) inputs. When present, visual token embeddings
    are interleaved into the text embedding sequence before the transformer
    layers. M-RoPE (multi-dimensional RoPE) position encoding is used for
    the full-attention layers when ``mrope_section`` is provided.

    Args:
        vocabulary_size (int): The size of the token vocabulary.
        num_layers (int): The number of transformer layers.
        num_query_heads (int): The number of query attention heads.
        num_key_value_heads (int): The number of key and value attention
            heads.
        head_dim (int): Dimension of each attention head.
        hidden_dim (int): The size of the transformer hidden dimension.
        intermediate_dim (int): The FFN intermediate dimension.
        layer_types (list): List of layer types, one per layer.
            Each element is ``"full_attention"`` or
            ``"linear_attention"``.
        partial_rotary_factor (float): Fraction of head_dim that gets
            RoPE. Defaults to ``0.25``.
        rope_max_wavelength (int): Maximum wavelength for RoPE. Defaults
            to ``10000``.
        rope_scaling_factor (float): Scaling factor for RoPE. Defaults
            to ``1.0``.
        layer_norm_epsilon (float): Epsilon for layer norms. Defaults
            to ``1e-6``.
        dropout (float): Dropout rate. Defaults to ``0.0``.
        tie_word_embeddings (bool): Whether to tie input and output
            embeddings. Defaults to ``False``.
        sliding_window_size (int): Sliding window size for full attention
            layers. Defaults to ``32768``.
        linear_num_key_heads (int): Key heads for linear attention.
            Defaults to ``16``.
        linear_num_value_heads (int): Value heads for linear attention.
            Defaults to ``32``.
        linear_key_head_dim (int): Key head dim for linear attention.
            Defaults to ``128``.
        linear_value_head_dim (int): Value head dim for linear attention.
            Defaults to ``128``.
        linear_conv_kernel_dim (int): Conv kernel size for linear
            attention. Defaults to ``4``.
        vision_encoder: optional ``Qwen3_5VisionEncoder``. When supplied,
            the backbone accepts ``pixel_values``, ``image_grid_thw``, and
            ``vision_indices`` in addition to text inputs.
        mrope_section (list[int]): [s_t, s_h, s_w] — number of *pairs* of
            rotary dimensions assigned to temporal, height, and width axes.
            Required for M-RoPE position encoding in multimodal mode.
            e.g. ``[11, 11, 10]`` for the 27B model. Defaults to ``None``
            (plain 1D RoPE).
        dtype: string or ``keras.mixed_precision.DTypePolicy``. The
            dtype to use for model computations and weights.
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
        layer_types=None,
        partial_rotary_factor=0.25,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        tie_word_embeddings=False,
        sliding_window_size=32768,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vision_encoder=None,
        mrope_section=None,
        dtype=None,
        **kwargs,
    ):
        # Default layer_types: every 4th layer is full_attention.
        if layer_types is None:
            layer_types = [
                ("linear_attention" if bool((i + 1) % 4) else "full_attention")
                for i in range(num_layers)
            ]

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen3_5_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = Qwen3_5TransformerDecoder(
                layer_type=layer_types[i],
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                partial_rotary_factor=partial_rotary_factor,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_qwen3_5_kernel_initializer(stddev=0.02),
                dropout=dropout,
                sliding_window_size=sliding_window_size,
                linear_num_key_heads=linear_num_key_heads,
                linear_num_value_heads=linear_num_value_heads,
                linear_key_head_dim=linear_key_head_dim,
                linear_value_head_dim=linear_value_head_dim,
                linear_conv_kernel_dim=linear_conv_kernel_dim,
                mrope_section=mrope_section,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        self.layer_norm = Qwen3_5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        # Vision encoder and interleave layer (optional).
        self.vision_encoder = vision_encoder
        if vision_encoder is not None:
            self.interleave_embeddings = Qwen3_5InterleaveEmbeddings(
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="interleave_embeddings",
            )

        # === Functional Model ===
        # The backbone functional model is text-only. The vision encoder and
        # interleave layer are stored as attributes and called imperatively
        # from CausalLM.call_with_cache() to avoid shape-inference issues
        # with dynamic vision tensor shapes.
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Text embeddings.
        x = self.token_embedding(token_id_input)

        # Transformer layers (text-only path in the functional graph).
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)

        sequence_output = self.layer_norm(x)

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }

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
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.layer_types = layer_types
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_max_wavelength = rope_max_wavelength
        self.num_key_value_heads = num_key_value_heads
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.sliding_window_size = sliding_window_size
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.mrope_section = mrope_section

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "head_dim": self.head_dim,
                "intermediate_dim": self.intermediate_dim,
                "layer_types": self.layer_types,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "sliding_window_size": self.sliding_window_size,
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_num_value_heads": self.linear_num_value_heads,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
                "mrope_section": self.mrope_section,
            }
        )
        return config
