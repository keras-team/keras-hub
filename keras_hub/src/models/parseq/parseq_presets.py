"""PARSeq preset configurations."""

backbone_presets = {
    "parseq_vit": {
        "metadata": {
            "description": (
                "Permuted autoregressive sequence (PARSeq) base "
                "model for scene text recognition"
            ),
            "params": 23_832_671,
            "path": "parseq",
        },
        "kaggle_handle": "kaggle://keras/parseq/keras/parseq_vit/1",
    }
}
