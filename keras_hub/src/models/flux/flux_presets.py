"""FLUX model preset configurations."""

presets = {
    "flux1_schnell": {
        "metadata": {
            "description": (
                "FLUX.1 [schnell] 12B rectified flow transformer, "
                "timestep-distilled for few-step sampling. Transformer "
                "(backbone) weights only: the text encoders and the VAE are "
                "not included, so text-to-image generation is not supported "
                "yet."
            ),
            "params": 11891178560,
            "path": "flux",
            "model_card": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        },
        "kaggle_handle": "kaggle://keras/flux/keras/flux1_schnell",
    },
}
