"""Qwen2.5-VL preset configurations."""

# Qwen releases Qwen2.5-VL only as instruction-tuned checkpoints.
# TODO: uncomment each entry once its Kaggle weights are uploaded.
backbone_presets = {}

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
