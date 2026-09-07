"""MuseGlimmer model preset configurations.

No presets are registered yet. Per CONTRIBUTING.md's post-merge process,
the Keras team uploads converted weights and registers the real preset
entry after this PR merges — the block below documents the expected shape
only and must stay commented out until then.
"""

# TODO: Fill in after weight conversion is complete and validated.
# backbone_cls = MuseGlimmerBackbone
# muse_glimmer_30b = {
#     "metadata": {
#         "description": (
#             "30B-parameter dense causal decoder with an attached "
#             "windowed-attention perception encoder, distilled from "
#             "Muse Spark for local agentic use."
#         ),
#         "params": 29_600_000_000,
#         "official_name": "MuseGlimmer",
#         "path": "muse_glimmer",
#     },
#     "kaggle_handle": "kaggle://keras/muse-glimmer/keras/muse_glimmer_30b/1",
# }

backbone_presets = {}
