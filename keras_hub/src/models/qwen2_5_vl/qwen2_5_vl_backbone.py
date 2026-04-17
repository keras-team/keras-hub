import keras
from keras import ops

from qwen2_5_vl_vision_encoder import Qwen2_5_VLVisionEncoder
from qwen2_5_vl_patch_merger import Qwen2_5_VLPatchMerger
from qwen2_5_vl_projector import Qwen2_5_VLVisionProjector
from qwen2_5_vl_decoder_stack import Qwen2_5_VLDecoderStack


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLBackbone(keras.Model):
    """
    Qwen2.5-VL Backbone.

    Combines a vision encoder, patch merger, vision projector, and a
    causal language model decoder stack. Supports text-only and multimodal
    (image + text) forward passes.

    Parameters
    ----------
    vocab_size : int
    hidden_size : int
    num_layers : int
    num_heads : int
    num_kv_heads : int
    intermediate_size : int
    vision_hidden_size : int
    vision_num_heads : int
    vision_intermediate_size : int
    vision_num_layers : int
    patch_size : int
    window_size : int
    rms_epsilon : float
    dropout : float
    vision_projector_intermediate_multiplier : int
        Multiplier for the vision projector intermediate layer size.
        Use 1 for 3B, 2 for 7B and 72B.
    dtype : str
        Compute dtype. Use "bfloat16" or "float16" for memory efficiency.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        vision_hidden_size,
        vision_num_heads,
        vision_intermediate_size,
        vision_num_layers,
        patch_size=14,
        window_size=8,
        rms_epsilon=1e-6,
        dropout=0.0,
        vision_projector_intermediate_multiplier=2,
        dtype="float32",
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_heads = vision_num_heads
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_layers = vision_num_layers
        self.patch_size = patch_size
        self.window_size = window_size
        self.rms_epsilon = rms_epsilon
        self.dropout = dropout
        self.vision_projector_intermediate_multiplier = (
            vision_projector_intermediate_multiplier
        )

        self.token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            name="token_embedding",
        )

        self.vision_encoder = Qwen2_5_VLVisionEncoder(
            hidden_size=vision_hidden_size,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            patch_size=patch_size,
            window_size=window_size,
            temporal_patch_size=2,
            global_layers=[],
            name="vision_encoder",
        )

        self.patch_merger = Qwen2_5_VLPatchMerger(
            name="patch_merger",
        )

        self.vision_projector = Qwen2_5_VLVisionProjector(
            vision_hidden_size=4 * vision_hidden_size,
            text_hidden_size=hidden_size,
            intermediate_multiplier=vision_projector_intermediate_multiplier,
            name="vision_projector",
        )

        self.decoder_stack = Qwen2_5_VLDecoderStack(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            rms_epsilon=rms_epsilon,
            dropout=dropout,
            name="decoder_stack",
        )

    def build(self, input_shape=None):
        self.token_embedding.build((None, None))
        self.built = True

    def _inject_vision_tokens(self, text_embeds, vision_embeds, image_mask):
        image_mask    = ops.cast(image_mask, "bool")
        batch_size    = ops.shape(text_embeds)[0]
        seq_len       = ops.shape(text_embeds)[1]
        hidden_dim    = ops.shape(text_embeds)[2]
        n_vision      = ops.shape(vision_embeds)[1]
        pad_len       = seq_len - n_vision
        padding       = ops.zeros((batch_size, pad_len, hidden_dim), dtype=text_embeds.dtype)
        vision_padded = ops.concatenate([vision_embeds, padding], axis=1)
        mask_expanded = ops.expand_dims(image_mask, axis=-1)
        return ops.where(mask_expanded, vision_padded, text_embeds)

    def call(self, inputs, training=False):
        token_ids      = inputs["token_ids"]
        pixel_values   = inputs.get("pixel_values",   None)
        image_mask     = inputs.get("image_mask",     None)
        attention_mask = inputs.get("attention_mask", None)

        hidden_states = self.token_embedding(token_ids)

        if pixel_values is not None:
            if image_mask is None:
                raise ValueError(
                    "`image_mask` is required when `pixel_values` are provided."
                )
            vision_outputs = self.vision_encoder(pixel_values, training=training)
            vision_tokens  = vision_outputs["vision_tokens"]
            grid           = vision_outputs["grid_thw"]
            vision_tokens  = self.patch_merger(
                vision_tokens, T=grid[0], H=grid[1], W=grid[2]
            )
            vision_tokens  = self.vision_projector(vision_tokens, training=training)
            hidden_states  = self._inject_vision_tokens(
                hidden_states, vision_tokens, image_mask
            )

        hidden_states = self.decoder_stack(
            hidden_states,
            attention_mask=attention_mask,
            training=training,
        )
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size":               self.vocab_size,
            "hidden_size":              self.hidden_size,
            "num_layers":               self.num_layers,
            "num_heads":                self.num_heads,
            "num_kv_heads":             self.num_kv_heads,
            "intermediate_size":        self.intermediate_size,
            "vision_hidden_size":       self.vision_hidden_size,
            "vision_num_heads":         self.vision_num_heads,
            "vision_intermediate_size": self.vision_intermediate_size,
            "vision_num_layers":        self.vision_num_layers,
            "patch_size":               self.patch_size,
            "window_size":              self.window_size,
            "rms_epsilon":              self.rms_epsilon,
            "dropout":                  self.dropout,
            "vision_projector_intermediate_multiplier":
                self.vision_projector_intermediate_multiplier,
            "dtype":                    self.dtype,
        })
        return config