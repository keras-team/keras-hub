"""LayoutLMv3 presets."""

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone
from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import LayoutLMv3Tokenizer

def layoutlmv3_base(
    *,
    load_weights=True,
    **kwargs,
):
    """Create a LayoutLMv3 base model.
    
    Args:
        load_weights: Whether to load pretrained weights.
        **kwargs: Additional keyword arguments.
        
    Returns:
        A tuple of (backbone, tokenizer).
    """
    backbone = LayoutLMv3Backbone(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=(112, 112),
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        **kwargs,
    )
    
    tokenizer = LayoutLMv3Tokenizer(
        vocabulary=None,  # Will be loaded from pretrained weights
        lowercase=True,
        strip_accents=True,
    )
    
    if load_weights:
        # TODO: Load pretrained weights from GCP bucket
        pass
    
    return backbone, tokenizer

def layoutlmv3_large(
    *,
    load_weights=True,
    **kwargs,
):
    """Create a LayoutLMv3 large model.
    
    Args:
        load_weights: Whether to load pretrained weights.
        **kwargs: Additional keyword arguments.
        
    Returns:
        A tuple of (backbone, tokenizer).
    """
    backbone = LayoutLMv3Backbone(
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=(112, 112),
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        **kwargs,
    )
    
    tokenizer = LayoutLMv3Tokenizer(
        vocabulary=None,  # Will be loaded from pretrained weights
        lowercase=True,
        strip_accents=True,
    )
    
    if load_weights:
        # TODO: Load pretrained weights from GCP bucket
        pass
    
    return backbone, tokenizer

# Dictionary mapping preset names to their corresponding functions
LAYOUTLMV3_PRESETS = {
    "layoutlmv3_base": layoutlmv3_base,
    "layoutlmv3_large": layoutlmv3_large,
} 