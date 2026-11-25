"""OpenAI Privacy Filter preset configurations."""

backbone_presets = {
    "openai_privacy_filter_en": {
        "metadata": {
            "description": (
                "8-layer encoder-only MoE token classifier (128 experts, "
                "top-4 routing) for English PII detection across 8 entity "
                "types and 33 BIOES labels."
            ),
            "params": 1399465712,
            "path": "openai_privacy_filter",
        },
        "kaggle_handle": (
            "kaggle://keras/openai-privacy-filter/keras/"
            "openai_privacy_filter_en/1"
        ),
    },
}
