"""ESM model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "esm2_t6_8M": {
        "metadata": {
            "description": (
                "6 transformer layers version of the ESM-2 protein language "
                "model, trained on the UniRef50 clustered protein sequence "
                "dataset."
            ),
            "params": 7_408_960,
            "path": "esm",
        },
        "kaggle_handle": "kaggle://keras/esm-2/keras/esm2_t6_8M/1",
    },
    "esm2_t12_35M": {
        "metadata": {
            "description": (
                "12 transformer layers version of the ESM-2 protein language "
                "model, trained on the UniRef50 clustered protein sequence "
                "dataset."
            ),
            "params": 33_269_280,
            "path": "esm",
        },
        "kaggle_handle": "kaggle://keras/esm-2/keras/esm2_t12_35M/1",
    },
    "esm2_t30_150M": {
        "metadata": {
            "description": (
                "30 transformer layers version of the ESM-2 protein language "
                "model, trained on the UniRef50 clustered protein sequence "
                "dataset."
            ),
            "params": 147_728_000,
            "path": "esm",
        },
        "kaggle_handle": "kaggle://keras/esm-2/keras/esm2_t30_150M/1",
    },
    "esm2_t33_650M": {
        "metadata": {
            "description": (
                "33 transformer layers version of the ESM-2 protein language "
                "model, trained on the UniRef50 clustered protein sequence "
                "dataset."
            ),
            "params": 649_400_320,
            "path": "esm",
        },
        "kaggle_handle": "kaggle://keras/esm-2/keras/esm2_t33_650M/1",
    },
}
