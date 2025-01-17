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
            "path": "stable_diffusion_3",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion3/keras/stable_diffusion_3_medium/4",
    },
    "stable_diffusion_3.5_medium": {
        "metadata": {
            "description": (
                "3 billion parameter, including CLIP L and CLIP G text "
                "encoders, MMDiT-X generative model, and VAE autoencoder. "
                "Developed by Stability AI."
            ),
            "params": 3371793763,
            "path": "stable_diffusion_3",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion3/keras/stable_diffusion_3.5_medium/1",
    },
    "stable_diffusion_3.5_large": {
        "metadata": {
            "description": (
                "9 billion parameter, including CLIP L and CLIP G text "
                "encoders, MMDiT generative model, and VAE autoencoder. "
                "Developed by Stability AI."
            ),
            "params": 9048410595,
            "path": "stable_diffusion_3",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion-3.5/keras/stable_diffusion_3.5_large/2",
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
            "path": "stable_diffusion_3",
        },
        "kaggle_handle": "kaggle://keras/stablediffusion-3.5/keras/stable_diffusion_3.5_large_turbo/2",
    },
}
