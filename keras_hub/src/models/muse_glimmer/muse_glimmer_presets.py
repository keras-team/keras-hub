"""MuseGlimmer model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "muse_glimmer_30b": {
        "metadata": {
            "description": (
                "30B-parameter dense causal decoder with an attached "
                "windowed-attention perception encoder, distilled from "
                "Muse Spark for local agentic use."
            ),
            "params": 29776626688,
            "path": "muse_glimmer",
        },
        "kaggle_handle": "kaggle://keras/muse-glimmer/keras/muse_glimmer_30b/1",
    },
    "muse_glimmer_30b_assistant": {
        "metadata": {
            "description": (
                "MuseGlimmer 30B DFlash Assistant model: 5-layer "
                "speculative-decoding drafter for the 30B model. Drafts a "
                "16-token block-diffusion denoised block per forward pass, "
                "conditioned on the target model's hidden states. This "
                "model must NOT be used standalone. It is designed "
                "exclusively as a draft model used from the target "
                "model's generation loop."
            ),
            "params": 2555985152,
            "path": "muse_glimmer",
        },
        "kaggle_handle": "kaggle://keras/muse-glimmer/keras/muse_glimmer_30b_assistant/1",
    },
}
