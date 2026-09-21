"""FLUX model preset configurations."""

presets = {
    "flux1_schnell": {
        "metadata": {
            "description": (
                "FLUX.1 [schnell] 12B rectified flow transformer for "
                "text-to-image generation, timestep-distilled for "
                "few-step sampling."
            ),
            "params": 11901408256,
            "path": "flux",
            "model_card": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        },
        "kaggle_handle": "kaggle://keras/flux/keras/flux1_schnell",
    },
}
