"""
Preset configurations for Qwen2.5-VL model variants.

Values sourced from official HuggingFace config.json files:
  - Qwen/Qwen2.5-VL-3B-Instruct
  - Qwen/Qwen2.5-VL-7B-Instruct
  - Qwen/Qwen2.5-VL-72B-Instruct

Only keys accepted by Qwen2_5_VLBackbone.__init__ are included.
"""

PRESETS = {
    "qwen2_5_vl_tiny": {
        "vocab_size":                          152064,
        "hidden_size":                         64,
        "num_layers":                          2,
        "num_heads":                           4,
        "num_kv_heads":                        2,
        "intermediate_size":                   128,
        "vision_hidden_size":                  64,
        "vision_num_heads":                    4,
        "vision_intermediate_size":            128,
        "vision_num_layers":                   2,
        "patch_size":                          14,
        "window_size":                         8,
        "vision_projector_intermediate_multiplier": 2,
    },
    "qwen2_5_vl_3b": {
        "vocab_size":                          151936,
        "hidden_size":                         2048,
        "num_layers":                          36,
        "num_heads":                           16,
        "num_kv_heads":                        2,
        "intermediate_size":                   11008,
        "vision_hidden_size":                  1280,
        "vision_num_heads":                    16,
        "vision_intermediate_size":            3420,
        "vision_num_layers":                   32,
        "patch_size":                          14,
        "window_size":                         112,
        "vision_projector_intermediate_multiplier": 1,
    },
    "qwen2_5_vl_7b": {
        "vocab_size":                          152064,
        "hidden_size":                         3584,
        "num_layers":                          28,
        "num_heads":                           28,
        "num_kv_heads":                        4,
        "intermediate_size":                   18944,
        "vision_hidden_size":                  1280,
        "vision_num_heads":                    16,
        "vision_intermediate_size":            3456,
        "vision_num_layers":                   32,
        "patch_size":                          14,
        "window_size":                         112,
        "vision_projector_intermediate_multiplier": 2,
    },
    "qwen2_5_vl_72b": {
        "vocab_size":                          152064,
        "hidden_size":                         8192,
        "num_layers":                          80,
        "num_heads":                           64,
        "num_kv_heads":                        8,
        "intermediate_size":                   29568,
        "vision_hidden_size":                  1280,
        "vision_num_heads":                    16,
        "vision_intermediate_size":            3456,
        "vision_num_layers":                   32,
        "patch_size":                          14,
        "window_size":                         112,
        "vision_projector_intermediate_multiplier": 2,
    },
}