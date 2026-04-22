"""BLIP-2 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "blip2_opt_2_7b": {
        "metadata": {
            "description": (
                "3.74 billion parameter BLIP-2 model with EVA-CLIP vision "
                "encoder (ViT, 985M), Q-Former (105M), language projection "
                "(2M), and OPT-2.7B language model (2.65B). Pretrained for "
                "image-text matching and visual question answering."
            ),
            "params": 3744737280,
            "path": "blip2",
        },
        "kaggle_handle": "kaggle://keras/blip2/keras/blip2_opt_2_7b/1",
    },
}
