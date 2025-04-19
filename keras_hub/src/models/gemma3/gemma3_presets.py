"""Gemma3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma3_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 26-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 999885952,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_1b/3",
    },
    "gemma3_instruct_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 26-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 999885952,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_1b/3",
    },
    "gemma3_4b_text": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 3880099328,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_4b_text/2",
    },
    "gemma3_instruct_4b_text": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 3880099328,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_4b_text/3",
    },
    "gemma3_12b_text": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 11765788416,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_12b_text/2",
    },
    "gemma3_instruct_12b_text": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 11765788416,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_12b_text/2",
    },
    "gemma3_27b_text": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 27009002240,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_27b_text/3",
    },
    "gemma3_instruct_27b_text": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 27009002240,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_27b_text/2",
    },
    "gemma3_4b": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, vision+text pretrained "
                "Gemma3 model."
            ),
            "params": 4299915632,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_4b/1",
    },
    "gemma3_instruct_4b": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, vision+text instruction-tuned "
                "Gemma3 model."
            ),
            "params": 4299915632,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_4b/1",
    },
    "gemma3_12b": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, vision+text pretrained "
                "Gemma3 model."
            ),
            "params": 12187079280,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_12b/1",
    },
    "gemma3_instruct_12b": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, vision+text instruction-tuned "
                "Gemma3 model."
            ),
            "params": 12187079280,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_12b/1",
    },
    "gemma3_27b": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, vision+text pretrained "
                "Gemma3 model."
            ),
            "params": 27432062576,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_27b/1",
    },
    "gemma3_instruct_27b": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, vision+text instruction-tuned "
                "Gemma3 model."
            ),
            "params": 27432062576,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_27b/1",
    },
}
