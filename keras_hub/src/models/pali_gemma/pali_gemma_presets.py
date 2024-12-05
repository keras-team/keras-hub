"""PaliGemma model preset configurations."""

import re

# Metadata for loading pretrained model weights.
backbone_presets = {
    "pali_gemma_3b_mix_224": {
        "metadata": {
            "description": (
                "image size 224, mix fine tuned, text sequence " "length is 256"
            ),
            "params": 2923335408,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_mix_224/3",
    },
    "pali_gemma_3b_mix_448": {
        "metadata": {
            "description": (
                "image size 448, mix fine tuned, text sequence length is 512"
            ),
            "params": 2924220144,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_mix_448/3",
    },
    "pali_gemma_3b_224": {
        "metadata": {
            "description": (
                "image size 224, pre trained, text sequence length is 128"
            ),
            "params": 2923335408,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_224/3",
    },
    "pali_gemma_3b_448": {
        "metadata": {
            "description": (
                "image size 448, pre trained, text sequence length is 512"
            ),
            "params": 2924220144,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_448/3",
    },
    "pali_gemma_3b_896": {
        "metadata": {
            "description": (
                "image size 896, pre trained, text sequence length " "is 512"
            ),
            "params": 2927759088,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_896/3",
    },
    # PaliGemma2
    "pali_gemma2_ft_docci_3b_448": {
        "metadata": {
            "description": (
                "3 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 26-layer Gemma2 2B lanuage "
                "model. This model has been fine-tuned on the DOCCI dataset "
                "for improved descriptions with fine-grained details."
            ),
            "params": 3032979696,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        # TODO: Rename `pali_gemma_2_ft_docci_3b_448` to `pali_gemma2_ft_docci_3b_448`
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma_2_ft_docci_3b_448/1",
    },
    "pali_gemma2_ft_docci_10b_448": {
        "metadata": {
            "description": (
                "10 billion parameter, 27-layer for SigLIP-So400m vision "
                "encoder and 42-layer Gemma2 9B lanuage model. This model has "
                "been fine-tuned on the DOCCI dataset for improved "
                "descriptions with fine-grained details."
            ),
            "params": 9663294192,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_ft_docci_10b_448/1",
    },
    "pali_gemma2_pt_3b_224": {
        "metadata": {
            "description": (
                "3 billion parameter, image size 224, 27-layer for "
                "SigLIP-So400m vision encoder and 26-layer Gemma2 2B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 3032094960,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_3b_224/1",
    },
    "pali_gemma2_pt_3b_448": {
        "metadata": {
            "description": (
                "3 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 26-layer Gemma2 2B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 3032979696,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_3b_448/1",
    },
    "pali_gemma2_pt_3b_896": {
        "metadata": {
            "description": (
                "3 billion parameter, image size 896, 27-layer for "
                "SigLIP-So400m vision encoder and 26-layer Gemma2 2B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 3036518640,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_3b_896/1",
    },
    "pali_gemma2_pt_10b_224": {
        "metadata": {
            "description": (
                "10 billion parameter, image size 224, 27-layer for "
                "SigLIP-So400m vision encoder and 42-layer Gemma2 9B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 9662409456,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_10b_224/1",
    },
    "pali_gemma2_pt_10b_448": {
        "metadata": {
            "description": (
                "10 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 42-layer Gemma2 9B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 9663294192,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_10b_448/1",
    },
    "pali_gemma2_pt_10b_896": {
        "metadata": {
            "description": (
                "10 billion parameter, image size 896, 27-layer for "
                "SigLIP-So400m vision encoder and 42-layer Gemma2 9B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 9666833136,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_10b_896/1",
    },
    "pali_gemma2_pt_28b_224": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 224, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 9662409456,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_28b_224/1",
    },
    "pali_gemma2_pt_28b_448": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 9663294192,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_28b_448/1",
    },
    "pali_gemma2_pt_28b_896": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 896, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 9666833136,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_28b_896/1",
    },
}

# Ensure compatibility with the official naming convention.
# pali_gemma2_[3|10|28b]_[variant]_[image_size]
compatible_preset_names = []
for preset_name in backbone_presets.keys():
    if re.match(r"pali_gemma2_(.+)_(.+)_(.+)_(.+)", preset_name):
        # Ex: pali_gemma2_ft_docci_3b_448 -> pali_gemma2_3b_ft_docci_448
        compatible_preset_names.append(
            (
                re.sub(
                    r"pali_gemma2_(.+)_(.+)_(.+)_(.+)",
                    r"pali_gemma2_\3_\1_\2_\4",
                    preset_name,
                ),
                preset_name,
            )
        )
    elif re.match(r"pali_gemma2_(.+)_(.+)_(.+)", preset_name):
        # Ex: pali_gemma2_pt_3b_224 -> pali_gemma2_3b_pt_224
        compatible_preset_names.append(
            (
                re.sub(
                    r"pali_gemma2_(.+)_(.+)_(.+)",
                    r"pali_gemma2_\2_\1_\3",
                    preset_name,
                ),
                preset_name,
            )
        )
for compatible_preset_name, preset_name in compatible_preset_names:
    backbone_presets[compatible_preset_name] = backbone_presets[preset_name]
