"""SmolVLM2 preset configurations."""

smolvlm2_presets = {
    "smolvlm2_256m_video_instruct": {
        "metadata": {
            "description": (
                "256 million parameter SmolVLM2 vision-language model with a "
                "30-layer SmolLM2 text decoder and a 12-layer SigLIP vision "
                "encoder (512x512 images). Instruction-tuned for image and "
                "video understanding."
            ),
            "params": 256484928,
            "path": "smolvlm2",
        },
        "kaggle_handle": (
            "kaggle://keras/smolvlm2/keras/smolvlm2_256m_video_instruct/1"
        ),
    },
    "smolvlm2_500m_video_instruct": {
        "metadata": {
            "description": (
                "500 million parameter SmolVLM2 vision-language model with a "
                "32-layer SmolLM2 text decoder and a 12-layer SigLIP vision "
                "encoder (512x512 images). Instruction-tuned for image and "
                "video understanding."
            ),
            "params": 507482304,
            "path": "smolvlm2",
        },
        "kaggle_handle": (
            "kaggle://keras/smolvlm2/keras/smolvlm2_500m_video_instruct/1"
        ),
    },
    "smolvlm2_2.2b_instruct": {
        "metadata": {
            "description": (
                "2.2 billion parameter SmolVLM2 vision-language model with a "
                "24-layer SmolLM2 text decoder and a 27-layer SigLIP vision "
                "encoder (384x384 images). Instruction-tuned for image and "
                "video understanding."
            ),
            "params": 2246784880,
            "path": "smolvlm2",
        },
        "kaggle_handle": (
            "kaggle://keras/smolvlm2/keras/smolvlm2_2.2b_instruct/1"
        ),
    },
}
