"""PaliGemma model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "pali_gemma_3b_mix_224": {
        "metadata": {
            "description": (
                "image size 224, mix fine tuned, text sequence length is 256"
            ),
            "params": 2923335408,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_mix_224/4",
    },
    "pali_gemma_3b_mix_448": {
        "metadata": {
            "description": (
                "image size 448, mix fine tuned, text sequence length is 512"
            ),
            "params": 2924220144,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_mix_448/4",
    },
    "pali_gemma_3b_224": {
        "metadata": {
            "description": (
                "image size 224, pre trained, text sequence length is 128"
            ),
            "params": 2923335408,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_224/4",
    },
    "pali_gemma_3b_448": {
        "metadata": {
            "description": (
                "image size 448, pre trained, text sequence length is 512"
            ),
            "params": 2924220144,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_448/4",
    },
    "pali_gemma_3b_896": {
        "metadata": {
            "description": (
                "image size 896, pre trained, text sequence length is 512"
            ),
            "params": 2927759088,
            "path": "pali_gemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_896/4",
    },
    # PaliGemma2
    "pali_gemma_2_ft_docci_3b_448": {
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma_2_ft_docci_3b_448/2",
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_ft_docci_10b_448/3",
    },
    "pali_gemma2_mix_3b_224": {
        "metadata": {
            "description": (
                "3 billion parameter, image size 224, 27-layer for "
                "SigLIP-So400m vision encoder and 26-layer Gemma2 2B lanuage "
                "model. This model has been fine-tuned on a wide range of "
                "vision-language tasks and domains."
            ),
            "params": 3032094960,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_mix_3b_224/2",
    },
    "pali_gemma2_mix_3b_448": {
        "metadata": {
            "description": (
                "3 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 26-layer Gemma2 2B lanuage "
                "model. This model has been fine-tuned on a wide range of "
                "vision-language tasks and domains."
            ),
            "params": 3032979696,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_mix_3b_448/2",
    },
    "pali_gemma2_mix_10b_224": {
        "metadata": {
            "description": (
                "10 billion parameter, image size 224, 27-layer for "
                "SigLIP-So400m vision encoder and 42-layer Gemma2 9B lanuage "
                "model. This model has been fine-tuned on a wide range of "
                "vision-language tasks and domains."
            ),
            "params": 9662409456,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_mix_10b_224/3",
    },
    "pali_gemma2_mix_10b_448": {
        "metadata": {
            "description": (
                "10 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 42-layer Gemma2 9B lanuage "
                "model. This model has been fine-tuned on a wide range of "
                "vision-language tasks and domains."
            ),
            "params": 9663294192,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_mix_10b_448/3",
    },
    "pali_gemma2_mix_28b_224": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 224, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been fine-tuned on a wide range of "
                "vision-language tasks and domains."
            ),
            "params": 27650192112,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_28b_mix_224/3",
    },
    "pali_gemma2_mix_28b_448": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been fine-tuned on a wide range of "
                "vision-language tasks and domains."
            ),
            "params": 27650192112,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_28b_mix_448/3",
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_3b_224/2",
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_3b_448/2",
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_3b_896/2",
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_10b_224/3",
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_10b_448/3",
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
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_10b_896/3",
    },
    "pali_gemma2_pt_28b_224": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 224, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 27650192112,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_28b_224/4",
    },
    "pali_gemma2_pt_28b_448": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 448, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 27650192112,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_28b_448/3",
    },
    "pali_gemma2_pt_28b_896": {
        "metadata": {
            "description": (
                "28 billion parameter, image size 896, 27-layer for "
                "SigLIP-So400m vision encoder and 46-layer Gemma2 27B lanuage "
                "model. This model has been pre-trained on a mixture of "
                "datasets."
            ),
            "params": 27650192112,
            "official_name": "PaliGemma2",
            "path": "pali_gemma2",
            "model_card": "https://www.kaggle.com/models/google/paligemma-2",
        },
        "kaggle_handle": "kaggle://keras/paligemma2/keras/pali_gemma2_pt_28b_896/3",
    },
}
