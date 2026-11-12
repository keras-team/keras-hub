"""Qwen2.5-VL preset configurations."""

# Qwen releases Qwen2.5-VL only as instruction-tuned checkpoints on HF
# (Qwen/Qwen2.5-VL-{3,7,32}B-Instruct); no base variants exist upstream.
# TODO: uncomment each entry once its Kaggle weights are uploaded.
backbone_presets = {}

# backbone_presets["qwen2_5_vl_3b_instruct"] = {
#     "metadata": {
#         "description": (
#             "3 billion parameter Qwen2.5-VL vision-language model. "
#             "Instruction-tuned for image and video understanding with "
#             "window-attention vision encoder and multimodal rotary "
#             "position embeddings."
#         ),
#         "params": 3754622976,
#         "path": "qwen2_5_vl",
#     },
#     "kaggle_handle": "kaggle://keras/qwen2_5-vl/keras/qwen2_5_vl_3b_instruct/1",
# }
# backbone_presets["qwen2_5_vl_7b_instruct"] = {
#     "metadata": {
#         "description": (
#             "7 billion parameter Qwen2.5-VL vision-language model. "
#             "Instruction-tuned for image and video understanding."
#         ),
#         "params": 8292166656,
#         "path": "qwen2_5_vl",
#     },
#     "kaggle_handle": "kaggle://keras/qwen2_5-vl/keras/qwen2_5_vl_7b_instruct/1",
# }
# backbone_presets["qwen2_5_vl_32b_instruct"] = {
#     "metadata": {
#         "description": (
#             "32 billion parameter Qwen2.5-VL vision-language model. "
#             "Instruction-tuned for image and video understanding."
#         ),
#         "params": 33452718336,
#         "path": "qwen2_5_vl",
#     },
#     "kaggle_handle": "kaggle://keras/qwen2_5-vl/keras/qwen2_5_vl_32b_instruct/1",
# }
