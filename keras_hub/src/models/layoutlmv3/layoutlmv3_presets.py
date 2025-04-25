"""LayoutLMv3 model preset configurations."""

backbone_presets = {
    "layoutlmv3_base": {
        "metadata": {
            "description": (
                "12-layer LayoutLMv3 model with visual backbone. "
                "Trained on IIT-CDIP dataset for document understanding."
            ),
            "params": 113000000,
            "path": "layoutlmv3",
        },
        "kaggle_handle": "kaggle://keras/layoutlmv3/keras/layoutlmv3_base/1",
    },
    "layoutlmv3_large": {
        "metadata": {
            "description": (
                "24-layer LayoutLMv3 model with multimodal (text + layout + image) "
                "understanding capabilities. Trained on IIT-CDIP, RVL-CDIP, "
                "FUNSD, CORD, SROIE, and DocVQA datasets."
            ),
            "params": 340787200,
            "path": "layoutlmv3",
        },
        "kaggle_handle": "kaggle://keras/layoutlmv3/keras/layoutlmv3_large/3",
    },
}
 