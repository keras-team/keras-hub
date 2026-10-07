"""MobileNetV5 preset configurations."""

backbone_presets = {
    "mobilenetv5_300m_enc_gemma3n": {
        "metadata": {
            "description": (
                "Lightweight 300M-parameter convolutional vision encoder used "
                "as the image backbone for Gemma 3n"
            ),
            "params": 294_284_096,
            "path": "mobilenetv5",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv5/keras/mobilenetv5_300m_enc_gemma3n/1",
    }
}
