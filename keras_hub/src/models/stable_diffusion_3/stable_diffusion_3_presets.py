"""StableDiffusion3 preset configurations."""

backbone_presets = {
    "stable_diffusion_3_medium": {
        "metadata": {
            "description": (
                "3 billion parameter, including CLIP L and CLIP G text "
                "encoders, MMDiT generative model, and VAE autoencoder. "
                "Developed by Stability AI."
            ),
            "params": 2987080931,
            "official_name": "StableDiffusion3",
            "path": "stable_diffusion_3",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion3/keras/stable_diffusion_3_medium/1",
    }
}
