"""SAM preset configurations."""

backbone_presets = {
    "sam_base_sa1b": {
        "metadata": {
            "description": ("The base SAM model trained on the SA1B dataset."),
            "params": 93735728,
            "path": "sam",
        },
        "kaggle_handle": "kaggle://keras/sam/keras/sam_base_sa1b/5",
    },
    "sam_large_sa1b": {
        "metadata": {
            "description": ("The large SAM model trained on the SA1B dataset."),
            "params": 641090864,
            "path": "sam",
        },
        "kaggle_handle": "kaggle://keras/sam/keras/sam_large_sa1b/5",
    },
    "sam_huge_sa1b": {
        "metadata": {
            "description": ("The huge SAM model trained on the SA1B dataset."),
            "params": 312343088,
            "path": "sam",
        },
        "kaggle_handle": "kaggle://keras/sam/keras/sam_huge_sa1b/5",
    },
}
