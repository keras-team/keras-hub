"""OPT model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "opt_125m_en": {
        "metadata": {
            "description": (
                "12-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 125237760,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/opt/keras/opt_125m_en/2",
    },
    # We skip the 350m checkpoint because it does not match the structure of
    # other checkpoints.
    "opt_1.3b_en": {
        "metadata": {
            "description": (
                "24-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 1315753984,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/opt/keras/opt_1.3b_en/2",
    },
    "opt_2.7b_en": {
        "metadata": {
            "description": (
                "32-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 2700000000,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/opt/keras/opt_2.7b_en/2",
    },
    "opt_6.7b_en": {
        "metadata": {
            "description": (
                "32-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 6700000000,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/opt/keras/opt_6.7b_en/2",
    },
}
