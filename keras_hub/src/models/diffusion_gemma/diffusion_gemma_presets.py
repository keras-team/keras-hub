"""DiffusionGemma model preset configurations."""

backbone_presets = {
    "diffusion_gemma_26b_a4b_it": {
        "metadata": {
            "description": (
                "DiffusionGemma 26B MoE instruction-tuned model. 25.2B total "
                "parameters (3.8B active), 30 layers, 128 experts."
            ),
            "params": 25200000000,
            "path": "diffusion_gemma",
        },
    },
}
