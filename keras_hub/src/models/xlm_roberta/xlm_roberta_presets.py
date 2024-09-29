"""XLM-RoBERTa model preset configurations."""

backbone_presets = {
    "xlm_roberta_base_multi": {
        "metadata": {
            "description": (
                "12-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 277450752,
            "official_name": "XLM-RoBERTa",
            "path": "xlm_roberta",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/xlmr/README.md",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_base_multi/2",
    },
    "xlm_roberta_large_multi": {
        "metadata": {
            "description": (
                "24-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 558837760,
            "official_name": "XLM-RoBERTa",
            "path": "xlm_roberta",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/xlmr/README.md",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_large_multi/2",
    },
}
