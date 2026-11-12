"""FLUX.2 model preset configurations."""

presets = {
    "flux2_klein_4b": {
        "metadata": {
            "description": (
                "FLUX.2 [klein] 4B rectified flow transformer for "
                "text-to-image generation and image editing, step-distilled "
                "for 4-step sampling. Conditioned on Qwen3-4B hidden states."
            ),
            "params": 3875544576,
            "path": "flux2",
            "model_card": (
                "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
            ),
        },
        "kaggle_handle": "kaggle://keras/flux2/keras/flux2_klein_4b",
    },
}
