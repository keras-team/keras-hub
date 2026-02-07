"""MetaCLIP 2 model preset configurations."""

# Metadata for loading pretrained model weights.
# These presets correspond to the MetaCLIP 2 Worldwide models from Meta AI.
# https://huggingface.co/collections/facebook/meta-clip-2
backbone_presets = {
    # ViT-H-14-quickgelu-worldwide (224 resolution)
    "metaclip_2_vit_huge_patch14_224": {
        "metadata": {
            "description": (
                "986 million parameter, 32-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 224x224. MetaCLIP 2 "
                "worldwide huge model (ViT-H-14-quickgelu-worldwide) trained on "  # noqa
                "29B seen pairs with QuickGELU activation."
            ),
            "params": 1858784002,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip_2/keras/metaclip_2_vit_huge_patch14_224/1",
    },
    # ViT-H-14-378-worldwide (378 resolution)
    "metaclip_2_vit_huge_patch14_378": {
        "metadata": {
            "description": (
                "986 million parameter, 32-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 378x378. MetaCLIP 2 "
                "worldwide huge model (ViT-H-14-378-worldwide) trained on "
                "29B seen pairs."
            ),
            "params": 1859389185,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip_2/keras/metaclip_2_vit_huge_patch14_378/1",
    },
    # ViT-bigG-14-worldwide (224 resolution)
    "metaclip_2_vit_giant_patch14_224": {
        "metadata": {
            "description": (
                "1.4 billion parameter, 40-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 224x224. MetaCLIP 2 "
                "worldwide giant model (ViT-bigG-14-worldwide) trained on "
                "29B seen pairs."
            ),
            "params": 3630409985,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip_2/keras/metaclip_2_vit_giant_patch14_224/1",
    },
    # ViT-bigG-14-378-worldwide (378 resolution)
    "metaclip_2_vit_giant_patch14_378": {
        "metadata": {
            "description": (
                "1.4 billion parameter, 40-layer for vision and 24-layer for "
                "text, patch size of 14, image resolution 378x378. MetaCLIP 2 "
                "worldwide giant model (ViT-bigG-14-378-worldwide) trained on "
                "29B seen pairs."
            ),
            "params": 3631197057,
            "path": "metaclip_2",
        },
        "kaggle_handle": "kaggle://keras/metaclip_2/keras/metaclip_2_vit_giant_patch14_378/1",
    },
}
