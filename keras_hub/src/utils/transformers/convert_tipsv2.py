"""Weight converter for TIPSv2 models.

Converts weights from the HF custom-code format (google/tipsv2-*) to
KerasHub's TIPSv2Backbone format.

Note: TIPSv2 uses custom code on HF (trust_remote_code=True), NOT the
standard HF Transformers model registry. Weight keys follow PyTorch
module paths from the HF repo's modeling_tips.py / image_encoder.py /
text_encoder.py.
"""

import numpy as np

from keras_hub.src.models.tipsv2.tipsv2_backbone import TIPSv2Backbone
from keras_hub.src.models.tipsv2.tipsv2_text_encoder import TIPSv2TextEncoder
from keras_hub.src.models.tipsv2.tipsv2_vision_encoder import (
    TIPSv2VisionEncoder,
)

backbone_cls = TIPSv2Backbone

# Vision function name → (embed_dim, depth, num_heads, mlp_ratio)
_VISION_CONFIGS = {
    "vit_base": (768, 12, 12, 4.0),
    "vit_large": (1024, 24, 16, 4.0),
    "vit_so400m": (1152, 27, 16, 4304 / 1152),
    "vit_giant2": (1536, 40, 24, 4.0),
}


def convert_backbone_config(transformers_config):
    """Build KerasHub config from the HF config.json."""
    # Vision architecture params (depth, num_heads, mlp_ratio) are not
    # stored in config.json — they are implicit in the vision_fn factory
    # function name, so we need a lookup table.
    vision_fn = transformers_config["vision_fn"]
    embed_dim, depth, num_heads, mlp_ratio = _VISION_CONFIGS[vision_fn]

    img_size = transformers_config["img_size"]
    patch_size = transformers_config["patch_size"]
    ffn_layer = transformers_config["ffn_layer"]
    init_values = transformers_config["init_values"]
    num_register_tokens = transformers_config["num_register_tokens"]

    text_hidden_size = transformers_config["text_hidden_size"]
    text_mlp_dim = transformers_config["text_mlp_dim"]
    text_num_heads = transformers_config["text_num_heads"]
    text_num_layers = transformers_config["text_num_layers"]
    vocab_size = transformers_config["vocab_size"]
    max_len = transformers_config["max_len"]
    temperature = transformers_config["temperature"]

    vision_encoder = TIPSv2VisionEncoder(
        patch_size=patch_size,
        hidden_dim=embed_dim,
        num_layers=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        init_values=init_values,
        num_register_tokens=num_register_tokens,
        ffn_layer=ffn_layer,
        image_shape=(img_size, img_size, 3),
    )
    text_encoder = TIPSv2TextEncoder(
        vocabulary_size=vocab_size,
        embedding_dim=text_hidden_size,
        hidden_dim=text_hidden_size,
        num_layers=text_num_layers,
        num_heads=text_num_heads,
        intermediate_dim=text_mlp_dim,
        max_sequence_length=max_len,
    )

    return {
        "vision_encoder": vision_encoder,
        "text_encoder": text_encoder,
        "temperature": temperature,
    }


def convert_weights(backbone, loader, transformers_config):
    """Port weights from HF safetensors to KerasHub backbone."""

    def port_dense(keras_layer, hf_prefix):
        """Port a Dense layer (transpose weight)."""
        loader.port_weight(
            keras_layer.kernel,
            f"{hf_prefix}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_layer.bias is not None:
            loader.port_weight(
                keras_layer.bias,
                f"{hf_prefix}.bias",
            )

    def port_ln(keras_layer, hf_prefix):
        """Port a LayerNormalization layer."""
        loader.port_weight(keras_layer.gamma, f"{hf_prefix}.weight")
        loader.port_weight(keras_layer.beta, f"{hf_prefix}.bias")

    vision = backbone.vision_encoder
    text = backbone.text_encoder

    # ── Vision Encoder ──────────────────────────────────────────────

    # CLS token.
    loader.port_weight(
        vision.embeddings.cls_token,
        "vision_encoder.cls_token",
    )

    # Position embeddings.
    loader.port_weight(
        vision.embeddings.position_embeddings,
        "vision_encoder.pos_embed",
    )

    # Register tokens.
    if vision.embeddings.register_tokens is not None:
        loader.port_weight(
            vision.embeddings.register_tokens,
            "vision_encoder.register_tokens",
        )

    # Patch embedding (Conv2D).
    loader.port_weight(
        vision.embeddings.patch_embedding.projection.kernel,
        "vision_encoder.patch_embed.proj.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        vision.embeddings.patch_embedding.projection.bias,
        "vision_encoder.patch_embed.proj.bias",
    )

    # Vision transformer blocks.
    for i, block in enumerate(vision.vision_blocks):
        prefix = f"vision_encoder.blocks.{i}"

        # Norm 1.
        port_ln(block.norm1, f"{prefix}.norm1")

        # Attention: fused QKV.
        port_dense(block.attn.qkv, f"{prefix}.attn.qkv")
        port_dense(block.attn.proj, f"{prefix}.attn.proj")

        # LayerScale 1.
        if block.ls1 is not None:
            loader.port_weight(block.ls1.gamma, f"{prefix}.ls1.gamma")

        # Norm 2.
        port_ln(block.norm2, f"{prefix}.norm2")

        # MLP / SwiGLU.
        ffn_layer = transformers_config.get("ffn_layer", "mlp")
        if ffn_layer == "swiglu":
            port_dense(block.mlp.w12, f"{prefix}.mlp.w12")
            port_dense(block.mlp.w3, f"{prefix}.mlp.w3")
        else:
            port_dense(block.mlp.fc1, f"{prefix}.mlp.fc1")
            port_dense(block.mlp.fc2, f"{prefix}.mlp.fc2")

        # LayerScale 2.
        if block.ls2 is not None:
            loader.port_weight(block.ls2.gamma, f"{prefix}.ls2.gamma")

    # Vision final norm.
    port_ln(vision.layernorm, "vision_encoder.norm")

    # ── Text Encoder ────────────────────────────────────────────────

    # Token embedding.
    loader.port_weight(
        text.token_embedding.embeddings,
        "text_encoder.token_embedding.weight",
    )

    # Text transformer blocks.

    for i, block in enumerate(text.text_blocks):
        prefix = f"text_encoder.transformer.resblocks.{i}"

        # LayerNorm 1.
        port_ln(block.ln_1, f"{prefix}.ln_1")

        # Attention: The HF text encoder uses nn.MultiheadAttention which
        # stores fused in_proj_weight (3*D, D) and in_proj_bias (3*D,),
        # plus out_proj.
        loader.port_weight(
            block.attn.in_proj.kernel,
            f"{prefix}.attn.in_proj_weight",
            hook_fn=lambda x, _: x.T,
        )
        loader.port_weight(
            block.attn.in_proj.bias,
            f"{prefix}.attn.in_proj_bias",
        )
        loader.port_weight(
            block.attn.out_proj.kernel,
            f"{prefix}.attn.out_proj.weight",
            hook_fn=lambda x, _: x.T,
        )
        loader.port_weight(
            block.attn.out_proj.bias,
            f"{prefix}.attn.out_proj.bias",
        )

        # LayerNorm 2.
        port_ln(block.ln_2, f"{prefix}.ln_2")

        # MLP.
        port_dense(block.mlp.c_fc, f"{prefix}.mlp.c_fc")
        port_dense(block.mlp.c_proj, f"{prefix}.mlp.c_proj")

    # Text final norm.
    port_ln(text.ln_final, "text_encoder.ln_final")
