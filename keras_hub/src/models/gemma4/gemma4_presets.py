"""Gemma4 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma4_2b": {
        "metadata": {
            "description": (
                "Gemma 4 E2B base model: 2.3B effective parameters (5.1B "
                "total with Per-Layer Embeddings), 35-layer, audio+vision+text "
                "pretrained Gemma4 model. The 'E' denotes effective parameters "
                "— PLE gives each decoder layer its own token embedding table, "
                "maximizing parameter efficiency for on-device deployment."
            ),
            "params": 5100000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_2b/1",
    },
    "gemma4_instruct_2b": {
        "metadata": {
            "description": (
                "Gemma 4 E2B instruction-tuned model: 2.3B effective parameters"
                " (5.1B total with Per-Layer Embeddings), 35-layer, "
                "audio+vision+text instruction-tuned Gemma4 model. The 'E' "
                "denotes effective parameters — PLE gives each decoder layer "
                "its own token embedding table, maximizing parameter efficiency"
                " for on-device deployment."
            ),
            "params": 5100000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_instruct_2b/1",
    },
    "gemma4_4b": {
        "metadata": {
            "description": (
                "Gemma 4 E4B base model: 4.5B effective parameters (7.9B "
                "total with Per-Layer Embeddings), 42-layer, audio+vision+text "
                "pretrained Gemma4 model. The 'E' denotes effective parameters "
                "— PLE gives each decoder layer its own token embedding table, "
                "maximizing parameter efficiency for on-device deployment."
            ),
            "params": 7900000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_4b/1",
    },
    "gemma4_instruct_4b": {
        "metadata": {
            "description": (
                "Gemma 4 E4B instruction-tuned model: 4.5B effective parameters"
                " (7.9B total with Per-Layer Embeddings), 42-layer, "
                "audio+vision+text instruction-tuned Gemma4 model. The 'E' "
                "denotes effective parameters — PLE gives each decoder layer "
                "its own token embedding table, maximizing parameter efficiency"
                " for on-device deployment."
            ),
            "params": 7900000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_instruct_4b/1",
    },
    "gemma4_26b_a4b": {
        "metadata": {
            "description": (
                "Gemma 4 26B A4B base model: Mixture-of-Experts (MoE) model "
                "with 26B total parameters and only 4B active parameters per "
                "forward pass, 30-layer, vision+text pretrained Gemma4 model. "
                "The 'A' denotes active parameters — by activating only a 4B "
                "subset during inference, this MoE model runs nearly as fast "
                "as a dense 4B model."
            ),
            "params": 26000000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_26b_a4b/1",
    },
    "gemma4_instruct_26b_a4b": {
        "metadata": {
            "description": (
                "Gemma 4 26B A4B instruction-tuned model: Mixture-of-Experts "
                "(MoE) model with 26B total parameters and only 4B active "
                "parameters per forward pass, 30-layer, vision+text "
                "instruction-tuned Gemma4 model. The 'A' denotes active "
                "parameters — by activating only a 4B subset during inference, "
                "this MoE model runs nearly as fast as a dense 4B model."
            ),
            "params": 26000000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_instruct_26b_a4b/1",
    },
    "gemma4_31b": {
        "metadata": {
            "description": (
                "Gemma 4 31B base model: 31B parameter, 60-layer, dense "
                "vision+text pretrained Gemma4 model. The dense model "
                "in the Gemma 4 family, offering maximum quality for "
                "deployments where inference speed is less of a constraint."
            ),
            "params": 31000000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_31b/1",
    },
    "gemma4_instruct_31b": {
        "metadata": {
            "description": (
                "Gemma 4 31B instruction-tuned model: 31B parameter, 60-layer, "
                "dense vision+text instruction-tuned Gemma4 model. The "
                "dense model in the Gemma 4 family, offering maximum quality "
                "for deployments where inference speed is less of a constraint."
            ),
            "params": 31000000000,
            "path": "gemma4",
        },
        "kaggle_handle": "kaggle://keras/gemma4/keras/gemma4_instruct_31b/1",
    },
}
