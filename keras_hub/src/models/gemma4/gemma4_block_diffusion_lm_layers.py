import keras
from keras import ops

from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization


class Gemma4BlockDiffusionSelfConditioning(keras.layers.Layer):
    """Self-conditioning layer for DiffusionGemma decoder.

    This layer refines the canvas embeddings at the start of each decoder
    denoising step by incorporating information from the logits predicted at
    the previous step.  It is the only parameter that exists in the decoder
    but NOT in the encoder (all other backbone weights are shared).

    On the first denoising step (when `prev_logits` is `None`), the layer
    is skipped and `canvas_embeds` is returned unchanged.

    Architecture:
        soft_embeds = softmax(prev_logits) @ embed_tokens_weight * embed_scale
        x           = pre_norm(soft_embeds)
        gate        = gelu(gate_proj(x), approximate=True)
        out         = down_proj(gate * up_proj(x))
        return post_norm(canvas_embeds + out)

    `pre_norm` has a learnable scale (standard RMSNorm).
    `post_norm` has NO learnable scale (pure L2 normalisation via
    `Gemma4VNorm`), matching the HF checkpoint which stores no
    `post_norm.weight` tensor.

    Args:
        hidden_dim: int. Dimensionality of the model's hidden representations.
        intermediate_dim: int. Intermediate dimension of the gated MLP.
        epsilon: float. Epsilon for RMS normalization layers. Defaults to
            `1e-6`.

    Call arguments:
        canvas_embeds: float tensor of shape `(B, canvas_length, hidden_dim)`.
            Raw canvas token embeddings from the current step.
        prev_logits: float tensor of shape `(B, canvas_length, vocab_size)`
            from the previous denoising step, or `None` on the first step.
        embed_tokens_weight: float tensor of shape `(vocab_size, hidden_dim)`.
            Shared token embedding matrix.
        embed_scale: float scalar. Embedding scale factor (typically
            `sqrt(hidden_dim)`).

    Returns:
        Float tensor of shape `(B, canvas_length, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon = epsilon

        self.pre_norm = RMSNormalization(
            epsilon=epsilon,
            dtype=self.dtype_policy,
            name="pre_norm",
        )
        self.gate_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="gate_proj",
        )
        self.up_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.down_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="down_proj",
        )
        # post_norm has no learnable scale — matches HF which stores no
        # post_norm.weight tensor for this module.
        self.post_norm = Gemma4VNorm(
            epsilon=epsilon,
            dtype=self.dtype_policy,
            name="post_norm",
        )

    def build(self, input_shape):
        # input_shape is the shape of
        # canvas_embeds: (B, canvas_length, hidden_dim)
        self.pre_norm.build(input_shape)
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)

        gate_out_shape = self.gate_proj.compute_output_shape(input_shape)
        self.down_proj.build(gate_out_shape)

        down_out_shape = self.down_proj.compute_output_shape(gate_out_shape)
        self.post_norm.build(down_out_shape)

        self.built = True

    def call(
        self, canvas_embeds, prev_logits, embed_tokens_weight, embed_scale
    ):
        if prev_logits is None:
            return self.post_norm(canvas_embeds)

        # Soft token embeddings: weighted combination of embedding rows.
        probs = ops.softmax(ops.cast(prev_logits, "float32"), axis=-1)
        embed_w = ops.cast(embed_tokens_weight, "float32")
        # (B, canvas_length, vocab_size) x (vocab_size, hidden_dim)
        soft_embeds = ops.matmul(probs, embed_w)
        soft_embeds = soft_embeds * ops.cast(embed_scale, "float32")
        soft_embeds = ops.cast(soft_embeds, self.compute_dtype)

        x = self.pre_norm(soft_embeds)
        gate = keras.activations.gelu(self.gate_proj(x), approximate=True)
        out = self.down_proj(gate * self.up_proj(x))

        return self.post_norm(canvas_embeds + out)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "epsilon": self.epsilon,
            }
        )
        return config
