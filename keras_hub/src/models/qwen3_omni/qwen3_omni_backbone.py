import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_moe.qwen3_moe_layernorm import Qwen3MoeLayerNorm
from keras_hub.src.models.qwen3_omni.qwen3_omni_decoder import (
    Qwen3OmniTransformerDecoder,
)


def _qwen3_omni_kernel_initializer(stddev=0.02):
    """Kernel initializer for Qwen3-Omni.
    """
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Qwen3OmniBackbone")
class Qwen3OmniBackbone(Backbone):
    """Qwen3-Omni Thinker backbone with MoE architecture.

    This backbone implements the Qwen3-Omni Thinker (comprehension) model,
    which is a Mixture-of-Experts (MoE) based multimodal model supporting
    text, audio, image, and video inputs.

    The architecture consists of:
    - Text embedding layer
    - Optional audio encoder (Whisper-style for speech/sound processing)
    - Optional vision encoder (ViT-style for image/video processing)
    - Embedding interleaving for multimodal fusion
    - 48 MoE-based transformer decoder blocks
    - RMSNorm output layer

    Model configuration (30B-A3B variant):
    - 128 experts, 8 experts per token
    - 48 decoder layers
    - 2048 hidden dimension
    - 32 query heads, 4 key-value heads (GQA)
    - 128 head dimension

    Note: This implements the Thinker component. The Talker (speech generation)
    and Code2Wav components are separate models.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer decoder layers.
        num_query_heads: int. The number of query attention heads.
        num_key_value_heads: int. The number of key/value attention heads.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of FFN layers.
        head_dim: int. The dimension of each attention head.
        num_experts: int. Number of experts in MoE layers. Defaults to 128.
        num_experts_per_tok: int. Number of experts activated per token (top-k).
            Defaults to 8.
        mrope_section: tuple. M-RoPE section dimensions (text, temporal, spatial).
            Must sum to head_dim // 2. Defaults to (24, 20, 20) for head_dim=128.
        rope_max_wavelength: int. Max wavelength for RoPE. Defaults to 1000000.
        rope_scaling_factor: float. Scaling factor for RoPE. Defaults to 1.0.
        rope_attention_scaling: float. Attention scaling for RoPE. Defaults to 1.0.
        layer_norm_epsilon: float. Epsilon for layer norm. Defaults to 1e-6.
        dropout: float. Dropout rate. Defaults to 0.0.
        tie_word_embeddings: bool. Whether to tie input/output embeddings.
            Defaults to False.
        sliding_window_size: int or None. Size of sliding attention window.
            Defaults to None (no sliding window).
        norm_topk_prob: bool. Whether to normalize top-k probabilities in routing.
            Defaults to True.
        decoder_sparse_step: int. Sparse step for MoE layers. Defaults to 1.
        router_aux_loss_coefficient: float. Auxiliary loss coefficient for load
            balancing. Defaults to 0.001.
        mlp_only_layers: list or None. Layer indices with dense FFN only.
            Defaults to None.
        audio_encoder: Optional audio encoder instance. Defaults to None.
        vision_encoder: Optional vision encoder instance. Defaults to None.
        dtype: string or DTypePolicy. Model dtype. Defaults to None.

    Examples:

    ```python
    import numpy as np
    import keras_hub

    # Text-only mode
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Randomly initialized Qwen3-Omni Thinker (30B-A3B config)
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
        rope_max_wavelength=1000000,
        dtype="bfloat16",
    )
    output = model(input_data)
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
        mrope_section=(24, 20, 20),
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
        dtype=None,
        **kwargs,
    ):
        # === Text Embedding Layer ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen3_omni_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        
        # === Multimodal Encoders (Optional) ===
        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder
        
        # TODO: Implement multimodal embedding interleaving
        # When audio/vision encoders are provided, embeddings need to be interleaved
        # with text embeddings based on special tokens and masks.
        
        # === MoE Transformer Decoder Layers ===
        if not mlp_only_layers:
            mlp_only_layers = []
        
        self.transformer_layers = []
        for i in range(num_layers):
            # Determine if this layer uses MoE (sparse) or dense FFN
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
        
        # === Output Layer Norm ===
        self.layer_norm = Qwen3MoeLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )
        
        # === Functional Model ===
        
        # Model inputs (text + optional multimodal)
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
        
        # TODO: Add multimodal inputs (audio, vision) to input spec
        # Will need audio features, vision features, and corresponding masks
        
        # Embed tokens
        x = self.token_embedding(token_id_input)
        
        # TODO: Implement multimodal fusion
        # When audio/vision encoders are present:
        # 1. Encode audio/vision inputs
        # 2. Interleave embeddings based on special tokens
        # 3. Generate appropriate M-RoPE position IDs for multimodal sequence
        
        # Pass through MoE transformer decoder layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                position_ids=None,  # Auto-generated in attention layer
                decoder_padding_mask=padding_mask_input,
            )
        
        # Final layer norm
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
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.moe_intermediate_dim = moe_intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
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
                "norm_topk_prob": self.norm_topk_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "router_aux_loss_coefficient": self.router_aux_loss_coefficient,
                "mlp_only_layers": self.mlp_only_layers,
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_attention_scaling": self.rope_attention_scaling,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "sliding_window_size": self.sliding_window_size,
                "audio_encoder": keras.layers.serialize(self.audio_encoder) if self.audio_encoder else None,
                "vision_encoder": keras.layers.serialize(self.vision_encoder) if self.vision_encoder else None,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Deserialize model config, including audio and vision encoders."""
        audio_encoder = config.pop("audio_encoder", None)
        vision_encoder = config.pop("vision_encoder", None)
        
        if audio_encoder is not None:
            audio_encoder = keras.layers.deserialize(audio_encoder)
        if vision_encoder is not None:
            vision_encoder = keras.layers.deserialize(vision_encoder)
        
        return cls(
            **config,
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
        )
    