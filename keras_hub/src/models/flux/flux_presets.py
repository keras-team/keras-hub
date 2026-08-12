"""FLUX model preset configurations."""

presets = {
    "flux1_schnell": {
        "metadata": {
            "description": "FLUX.1 [schnell] text-to-image model.",
            "params": 11891885120,
            "official_name": "FLUX.1 [schnell]",
            "path": "flux",
            "model_card": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        },
        "kaggle_handle": "kaggle://<org>/flux/keras/flux1_schnell",
    },
    "flux1_dev": {
        "metadata": {
            "description": "FLUX.1 [dev] text-to-image model.",
            "params": 11891885120,
            "official_name": "FLUX.1 [dev]",
            "path": "flux",
            "model_card": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        },
        "kaggle_handle": "kaggle://<org>/flux/keras/flux1_dev",
    },
}
