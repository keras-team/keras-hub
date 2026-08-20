"""DiffusionGemma model preset configurations."""

backbone_presets = {
    "diffusion_gemma_26b_a4b_it": {
        "metadata": {
            "description": (
                "DiffusionGemma 26B MoE instruction-tuned model. 25.2B total "
                "parameters (3.8B active), 30 layers, 128 experts."
            ),
            "params": 23823778864,
            "path": "diffusion_gemma",
        },
        "kaggle_handle": "kaggle://keras/diffusion_gemma/keras/diffusion_gemma_26b_a4b_it/1",
    },
}
