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
    # "siglip_so400m_patch14_224": {
    #     "metadata": {
    #         "description": (
    #             "877 million parameter, image size 224, pre-trained on WebLi."
    #         ),
    #         "params": 877360578,
    #         "official_name": "SigLIP",
    #         "path": "siglip",
    #         "model_card": "https://www.kaggle.com/models/kerashub/siglip",
    #     },
    #     "kaggle_handle": "kaggle://kerashub/siglip/keras/siglip_so400m_patch14_224/2",
    # },
}
