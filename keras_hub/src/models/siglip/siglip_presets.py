"""SigLIP model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "siglip_base_patch16_224": {
        "metadata": {
            "description": (
                "200 million parameter, image size 224, pre-trained on WebLi."
            ),
            "params": 203156230,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_base_patch16_224/2",
    },
    "siglip_base_patch16_256": {
        "metadata": {
            "description": (
                "200 million parameter, image size 256, pre-trained on WebLi."
            ),
            "params": 203202370,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_base_patch16_256/1",
    },
    "siglip_base_patch16_384": {
        "metadata": {
            "description": (
                "200 million parameter, image size 384, pre-trained on WebLi."
            ),
            "params": 203448450,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_base_patch16_384/1",
    },
    "siglip_base_patch16_512": {
        "metadata": {
            "description": (
                "200 million parameter, image size 512, pre-trained on WebLi."
            ),
            "params": 203792962,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_base_patch16_512/1",
    },
    "siglip_large_patch16_256": {
        "metadata": {
            "description": (
                "652 million parameter, image size 256, pre-trained on WebLi."
            ),
            "params": 652151106,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_large_patch16_256/1",
    },
    "siglip_large_patch16_384": {
        "metadata": {
            "description": (
                "652 million parameter, image size 384, pre-trained on WebLi."
            ),
            "params": 652479106,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_large_patch16_384/1",
    },
    "siglip_so400m_patch14_224": {
        "metadata": {
            "description": (
                "877 million parameter, image size 224, "
                "shape-optimized version, pre-trained on WebLi."
            ),
            "params": 877360578,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_so400m_patch14_224/2",
    },
    "siglip_so400m_patch14_384": {
        "metadata": {
            "description": (
                "877 million parameter, image size 384, "
                "shape-optimized version, pre-trained on WebLi."
            ),
            "params": 877961291,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_so400m_patch14_384/1",
    },
    "siglip_so400m_patch16_256_i18n": {
        "metadata": {
            "description": (
                "1.1 billion parameter, image size 256, "
                "shape-optimized version, pre-trained on WebLi."
            ),
            "params": 1128759282,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_so400m_patch16_256_i18n/1",
    },
    "siglip_base_patch16_256_multilingual": {
        "metadata": {
            "description": (
                "370 million parameter, image size 256, pre-trained on WebLi."
            ),
            "params": 370626370,
            "official_name": "SigLIP",
            "path": "siglip",
            "model_card": "https://www.kaggle.com/models/kerashub/siglip",
        },
        "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_base_patch16_256_multilingual/1",
    },
}
