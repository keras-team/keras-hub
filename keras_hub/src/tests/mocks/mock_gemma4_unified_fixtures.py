from keras_hub.src.models.gemma4_unified.gemma4_unified_image_converter import (
    Gemma4UnifiedImageConverter,
)


def raw_image_preprocessor_kwargs():
    """Raw-image preprocessor configuration shared by the unified tests.

    `patch_size=4` with `pooling_kernel_size=3` gives 432-dim patches, four
    soft tokens per image, two images per prompt and a 64-token sequence.
    `gemma4_unified_causal_lm_test.py` and
    `gemma4_unified_causal_lm_preprocessor_test.py` build their raw-image
    preprocessor from this dict, so the converter and preprocessor side is
    written once. The backbone side of the LM test (`model_patch_size=12`,
    `num_soft_tokens=4`) is still set there and must match these numbers.
    """
    return {
        "image_converter": Gemma4UnifiedImageConverter(
            image_size=(16, 16),
            patch_size=4,
            pooling_kernel_size=3,
            max_soft_tokens=4,
        ),
        "sequence_length": 64,
        "max_images_per_prompt": 2,
        "num_vision_tokens_per_image": 4,
    }
