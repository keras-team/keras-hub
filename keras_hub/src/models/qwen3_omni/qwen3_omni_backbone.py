import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_moe.qwen3_moe_layernorm import Qwen3MoeLayerNorm
from keras_hub.src.models.qwen3_omni.qwen3_omni_decoder import (
    Qwen3OmniTransformerDecoder,
)


def _qwen3_omni_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Qwen3OmniBackbone")
class Qwen3OmniBackbone(Backbone):
    """Qwen3-Omni multimodal Transformer backbone.

    This backbone implements the base Transformer network for the Qwen3-Omni
    model. It includes embedding lookups and transformer layers with a Mixture
    of Experts (MoE) architecture, using Multimodal Rotary Position Embedding
    (M-RoPE) for multimodal fusion. Audio/vision encoders can be optionally
    provided for multimodal operation.

    The default constructor gives a fully customizable, randomly initialized
    Qwen3-Omni model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_query_heads: int. The number of heads for the query projections in
            the attention layer.
        num_key_value_heads: int. The number of heads for the key and value
            projections in the attention layer.
        hidden_dim: int. The size of the transformer hidden state at the end of
            each transformer layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the feedforward network for each transformer.
        moe_intermediate_dim: int. The intermediate dimension for each expert
            in the MoE feedforward network.
        head_dim: int. The size of each attention head.
        num_experts: int. The number of experts in each MoE layer.
        num_experts_per_tok: int. The number of top experts to select for each
            token in the MoE layer.
        mrope_section: tuple. M-RoPE section dimensions
            (text, temporal, spatial). Must sum to head_dim // 2.
        rope_max_wavelength: int. Max wavelength for RoPE.
        rope_scaling_factor: float. Scaling factor for RoPE.
        rope_attention_scaling: float. Attention scaling for RoPE.
        layer_norm_epsilon: float. The epsilon value used for every layer norm
            in the transformer model.
        dropout: float. Dropout probability for the transformer encoder.
        tie_word_embeddings: bool. Whether to tie input/output embeddings.
        sliding_window_size: int or None. Size of sliding attention window.
        norm_topk_prob: bool. Whether to normalize top-k probabilities.
        decoder_sparse_step: int. Sparse step for MoE layers.
        router_aux_loss_coefficient: float. Auxiliary loss coefficient.
        mlp_only_layers: list of int or None. Layers to use dense FFN instead
            of MoE.
        audio_encoder: Qwen3OmniAudioEncoder or None. Pre-instantiated audio
            encoder.
        vision_encoder: Qwen3OmniVisionEncoder or None. Pre-instantiated vision
            encoder.
        image_token_id: int. Token ID for image placeholders.
        video_token_id: int. Token ID for video placeholders.
        audio_token_id: int. Token ID for audio placeholders.
        dtype: str or `keras.mixed_precision.DTypePolicy`. The dtype to use for
            the model's computations and weights.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Randomly initialized Qwen3-Omni decoder with custom config.
    model = keras_hub.models.Qwen3OmniBackbone(
        vocabulary_size=152064,
        num_layers=48,
        num_query_heads=32,
        num_key_value_heads=4,
        hidden_dim=2048,
        intermediate_dim=768,
        moe_intermediate_dim=768,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=8,
    )
    model(input_data)
    ```
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
        audio_encoder=None,
        vision_encoder=None,
        image_token_id=None,
        video_token_id=None,
        audio_token_id=None,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen3_omni_kernel_initializer(stddev=0.01),
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
                kernel_initializer=_qwen3_omni_kernel_initializer(stddev=0.02),
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

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                position_ids=None,
                decoder_padding_mask=padding_mask_input,
            )
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder

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
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id

    def call(self, inputs, training=False):
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        audio_features = inputs.get("audio_features", None)
        pixel_values = inputs.get("pixel_values", None)
        grid_thw = inputs.get("grid_thw", None)

        x = self._compute_embeddings(
            token_ids, audio_features, pixel_values, grid_thw
        )
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                position_ids=None,
                decoder_padding_mask=padding_mask,
                training=training,
            )
        return self.layer_norm(x)

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
                "router_aux_loss_coefficient": self.router_aux_loss_coefficient,
                "mlp_only_layers": self.mlp_only_layers,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "audio_token_id": self.audio_token_id,
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

    def _compute_embeddings(
        self,
        token_ids,
        audio_features=None,
        pixel_values=None,
        grid_thw=None,
    ):
        inputs_embeds = self.token_embedding(token_ids)

        if audio_features is not None and self.audio_encoder is not None:
            audio_embeds = self.audio_encoder(
                {"input_features": audio_features}
            )
            audio_mask = ops.equal(
                ops.cast(token_ids, "int32"), self.audio_token_id
            )
            inputs_embeds = self._masked_scatter(
                inputs_embeds, audio_mask, audio_embeds
            )

        if pixel_values is not None and self.vision_encoder is not None:
            vision_outputs = self.vision_encoder(
                {"pixel_values": pixel_values, "grid_thw": grid_thw}
            )
            visual_embeds = vision_outputs["pooler_output"]
            image_mask = ops.equal(
                ops.cast(token_ids, "int32"), self.image_token_id
            )
            video_mask = ops.equal(
                ops.cast(token_ids, "int32"), self.video_token_id
            )
            visual_mask = ops.logical_or(image_mask, video_mask)
            inputs_embeds = self._masked_scatter(
                inputs_embeds, visual_mask, visual_embeds
            )

        return inputs_embeds

    def _masked_scatter(self, target, mask, source):
        """Replace embeddings at masked positions with source embeddings."""
        mask_expanded = ops.cast(ops.expand_dims(mask, -1), target.dtype)
        mask_int = ops.cast(mask, "int32")
        cumsum = ops.cumsum(mask_int, axis=1)
        source_indices = ops.maximum(cumsum - 1, 0)
        source_indices_expanded = ops.expand_dims(source_indices, -1)
        scattered_values = ops.take_along_axis(
            source, source_indices_expanded, axis=1
        )
        result = target * (1 - mask_expanded) + scattered_values * mask_expanded
        return result
