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
            "model_card": "https://huggingface.co/stabilityai/stable-diffusion-3-medium",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion3/keras/stable_diffusion_3_medium/3",
    },
    "stable_diffusion_3.5_large": {
        "metadata": {
            "description": (
                "9 billion parameter, including CLIP L and CLIP G text "
                "encoders, MMDiT generative model, and VAE autoencoder. "
                "Developed by Stability AI."
            ),
            "params": 9048410595,
            "official_name": "StableDiffusion3",
            "path": "stable_diffusion_3",
            "model_card": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion-3.5/keras/stable_diffusion_3.5_large/1",
    },
    "stable_diffusion_3.5_large_turbo": {
        "metadata": {
            "description": (
                "9 billion parameter, including CLIP L and CLIP G text "
                "encoders, MMDiT generative model, and VAE autoencoder. "
                "A timestep-distilled version that eliminates classifier-free "
                "guidance and uses fewer steps for generation. "
                "Developed by Stability AI."
            ),
            "params": 9048410595,
            "official_name": "StableDiffusion3",
            "path": "stable_diffusion_3",
            "model_card": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion-3.5/keras/stable_diffusion_3.5_large/1",
    },
}
