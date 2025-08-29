# Import LayoutLMv3 components with error handling for backend compatibility
try:
    from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
        LayoutLMv3Backbone,
    )
except ImportError as e:
    # Graceful degradation for missing dependencies
    LayoutLMv3Backbone = None
    import warnings

    warnings.warn(f"LayoutLMv3Backbone import failed: {e}")

try:
    from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import (
        LayoutLMv3Tokenizer,
    )
except ImportError as e:
    # Graceful degradation for missing dependencies
    LayoutLMv3Tokenizer = None
    import warnings

    warnings.warn(f"LayoutLMv3Tokenizer import failed: {e}")

from keras_hub.src.utils.preset_utils import register_presets

# Only register presets if classes loaded successfully
if LayoutLMv3Backbone is not None:
    try:
        # Register presets if they exist
        backbone_presets = {}  # Empty for now - will be populated when presets are added
        register_presets(backbone_presets, LayoutLMv3Backbone)
    except Exception as e:
        import warnings

        warnings.warn(f"Failed to register LayoutLMv3 presets: {e}")
