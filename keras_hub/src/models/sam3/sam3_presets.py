"""SAM3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "sam3_pcs": {
        "metadata": {
            "description": (
                "30 million parameter Promptable Concept Segmentation (PCS) "
                "SAM model."
            ),
            "params": 30000000,
            "path": "sam3",
        },
        "kaggle_handle": "kaggle://keras/sam3/keras/sam3_pcs/1",
    },
}
