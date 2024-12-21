"""XLM-RoBERTa model preset configurations."""

backbone_presets = {
    "xlm_roberta_base_multi": {
        "metadata": {
            "description": (
                "12-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 277450752,
            "path": "xlm_roberta",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_base_multi/3",
    },
    "xlm_roberta_large_multi": {
        "metadata": {
            "description": (
                "24-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 558837760,
            "path": "xlm_roberta",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_large_multi/3",
    },
}
