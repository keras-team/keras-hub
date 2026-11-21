"""RWKV7 model preset configurations."""

backbone_presets = {
    "rwkv7_g1a_0.1b_en": {
        "metadata": {
            "description": (
                "150 million parameter RWKV7 model. Optimized for edge "
                "devices and mobile deployment."
            ),
            "params": 150000000,
            "path": "rwkv7",
        },
        "kaggle_handle": "kaggle://keras/rwkv7/keras/rwkv7_g1a_0.1b/1",
    },
    "rwkv7_g1a_0.3b_en": {
        "metadata": {
            "description": (
                "400 million parameter RWKV7 model. Small variant balancing "
                "speed and instruction following."
            ),
            "params": 400000000,
            "path": "rwkv7",
        },
        "kaggle_handle": "kaggle://keras/rwkv7/keras/rwkv7_g1a_0.3b/1",
    },
}
