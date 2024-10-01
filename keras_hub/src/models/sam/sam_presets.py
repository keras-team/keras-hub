"""SAM preset configurations."""

backbone_presets = {
    "sam_base_sa1b": {
        "metadata": {
            "description": ("The base SAM model trained on the SA1B dataset."),
            "params": 93735728,
            "official_name": "SAMImageSegmenter",
            "path": "sam",
            "model_card": "https://arxiv.org/abs/2304.02643",
        },
        "kaggle_handle": "kaggle://kerashub/sam/keras/sam_base_sa1b/2",
    },
    "sam_large_sa1b": {
        "metadata": {
            "description": ("The large SAM model trained on the SA1B dataset."),
            "params": 641090864,
            "official_name": "SAMImageSegmenter",
            "path": "sam",
            "model_card": "https://arxiv.org/abs/2304.02643",
        },
        "kaggle_handle": "kaggle://kerashub/sam/keras/sam_large_sa1b/2",
    },
    "sam_huge_sa1b": {
        "metadata": {
            "description": ("The huge SAM model trained on the SA1B dataset."),
            "params": 312343088,
            "official_name": "SAMImageSegmenter",
            "path": "sam",
            "model_card": "https://arxiv.org/abs/2304.02643",
        },
        "kaggle_handle": "kaggle://kerashub/sam/keras/sam_huge_sa1b/2",
    },
}
