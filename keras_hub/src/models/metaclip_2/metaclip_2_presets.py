"""MetaCLIP 2 model preset configurations."""

backbone_presets = {
    "metaclip_2_vit_huge_patch14_224": {
        "metadata": {
            "description": (
                "2 billion parameter, 32-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 224x224. MetaCLIP 2 "
                "worldwide huge model (ViT-H-14-quickgelu-worldwide) trained on "  # noqa
                "29B seen pairs with QuickGELU activation."
            ),
            "params": 1858783745,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip2/keras/metaclip_2_vit_huge_patch14_224/1",
    },
    "metaclip_2_vit_huge_patch14_378": {
        "metadata": {
            "description": (
                "2 billion parameter, 32-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 378x378. MetaCLIP 2 "
                "worldwide huge model (ViT-H-14-378-worldwide) trained on "
                "29B seen pairs."
            ),
            "params": 1859389185,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip2/keras/metaclip_2_vit_huge_patch14_378/1",
    },
    "metaclip_2_vit_giant_patch14_224": {
        "metadata": {
            "description": (
                "4 billion parameter, 40-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 224x224. MetaCLIP 2 "
                "worldwide giant model (ViT-bigG-14-worldwide) trained on "
                "29B seen pairs."
            ),
            "params": 3630409985,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip2/keras/metaclip_2_vit_giant_patch14_224/1",
    },
    "metaclip_2_vit_giant_patch14_378": {
        "metadata": {
            "description": (
                "4 billion parameter, 40-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 378x378. MetaCLIP 2 "
                "worldwide giant model (ViT-bigG-14-378-worldwide) trained on "
                "29B seen pairs."
            ),
            "params": 3631197057,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip2/keras/metaclip_2_vit_giant_patch14_378/1",
    },
}
