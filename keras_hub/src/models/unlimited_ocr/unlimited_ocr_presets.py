"""Unlimited-OCR model preset configurations."""

backbone_presets = {
    "unlimited_ocr": {
        "metadata": {
            "description": (
                "3.3 billion parameter, 12-layer, document OCR model. It has "
                "a SAM and CLIP vision encoder and a mixture of experts text "
                "decoder with 64 routed experts."
            ),
            "params": 3335504128,
            "path": "unlimited_ocr",
        },
        "kaggle_handle": "kaggle://keras/unlimited_ocr/keras/unlimited_ocr/1",
    },
}
