"""Mistral3 model preset configurations."""

# Metadata for loading pretrained model weights.
# TODO: fill in `params` and `kaggle_handle` once these presets are uploaded.
backbone_presets = {
    "mistral_small_3.1_24b_base_2503_en": {
        "metadata": {
            "description": (
                "Mistral Small 3.1 24B base model (text + Pixtral vision "
                "encoder), converted from "
                "mistralai/Mistral-Small-3.1-24B-Base-2503."
            ),
            "params": None,
            "path": "mistral3",
        },
        "kaggle_handle": None,
    },
    "mistral_small_3.1_24b_instruct_2503_en": {
        "metadata": {
            "description": (
                "Mistral Small 3.1 24B instruct model (text + Pixtral "
                "vision encoder), converted from "
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503."
            ),
            "params": None,
            "path": "mistral3",
        },
        "kaggle_handle": None,
    },
    "mistral_small_3.2_24b_instruct_2506_en": {
        "metadata": {
            "description": (
                "Mistral Small 3.2 24B instruct model (text + Pixtral "
                "vision encoder), converted from "
                "mistralai/Mistral-Small-3.2-24B-Instruct-2506."
            ),
            "params": None,
            "path": "mistral3",
        },
        "kaggle_handle": None,
    },
}
