"""DINOV2 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "dinov2_small": {
        "metadata": {
            "description": (
                "Vision Transformer (small-sized model) trained using DINOv2."
            ),
            "params": 22_582_656,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_small/1",
    },
    "dinov2_base": {
        "metadata": {
            "description": (
                "Vision Transformer (base-sized model) trained using DINOv2."
            ),
            "params": 87_632_640,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_base/1",
    },
    "dinov2_large": {
        "metadata": {
            "description": (
                "Vision Transformer (large-sized model) trained using DINOv2."
            ),
            "params": 305_771_520,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_large/1",
    },
    "dinov2_giant": {
        "metadata": {
            "description": (
                "Vision Transformer (giant-sized model) trained using DINOv2."
            ),
            "params": 1_138_585_088,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_giant/1",
    },
    "dinov2_with_registers_small": {
        "metadata": {
            "description": (
                "Vision Transformer (small-sized model) trained using DINOv2, "
                "with registers."
            ),
            "params": 22_584_192,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_with_registers_small/1",
    },
    "dinov2_with_registers_base": {
        "metadata": {
            "description": (
                "Vision Transformer (base-sized model) trained using DINOv2, "
                "with registers."
            ),
            "params": 87_635_712,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_with_registers_base/1",
    },
    "dinov2_with_registers_large": {
        "metadata": {
            "description": (
                "Vision Transformer (large-sized model) trained using DINOv2, "
                "with registers."
            ),
            "params": 305_775_616,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_with_registers_large/1",
    },
    "dinov2_with_registers_giant": {
        "metadata": {
            "description": (
                "Vision Transformer (giant-sized model) trained using DINOv2, "
                "with registers."
            ),
            "params": 1_138_591_232,
            "path": "dinov2",
        },
        "kaggle_handle": "kaggle://keras/dinov2/keras/dinov2_with_registers_giant/1",
    },
}
