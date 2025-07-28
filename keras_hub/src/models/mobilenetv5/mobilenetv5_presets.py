# keras_hub/src/models/mobilenetv5/mobilenetv5_presets.py

"""MobileNetV5 preset configurations."""

backbone_presets_base = {
    "mobilenetv5_base": {
        "metadata": {
            "description": (
                "Base MobileNetV5 model. This configuration is designed for "
                "demonstration and has not been pre-trained."
            ),
            "params": 0,  # Update with actual param count
            "path": "mobilenetv5",
        },
        "config": {
            "block_args": [
                # Stage 0: 128x128 in
                [
                    "er_r1_k3_s2_e4_c128",
                    "er_r2_k3_s1_e4_c128",
                ],
                # Stage 1: 256x256 in
                [
                    "uir_r1_a3_k5_s2_e6_c256",
                    "uir_r4_a5_k0_s1_e4_c256",
                ],
                # Stage 2: 640x640 in
                [
                    "uir_r1_a5_k5_s2_e6_c512",
                    "uir_r3_a5_k0_s1_e4_c512",
                    "mqa_r1_k3_h8_s2_d64_c512",
                    "uir_r1_a0_k0_s1_e2_c512",
                    "mqa_r1_k3_h8_s2_d64_c512",
                    "uir_r1_a0_k0_s1_e2_c512",
                    "mqa_r1_k3_h8_s2_d64_c512",
                    "uir_r1_a0_k0_s1_e2_c512",
                    "mqa_r1_k3_h8_s2_d64_c512",
                    "uir_r1_a0_k0_s1_e2_c512",
                    "mqa_r1_k3_h8_s2_d64_c512",
                    "uir_r1_a0_k0_s1_e2_c512",
                    "mqa_r1_k3_h8_s2_d64_c512",
                    "uir_r1_a0_k0_s1_e2_c512",
                ],
                # Stage 3: 1280x1280 in
                [
                    "uir_r1_a5_k5_s2_e6_c1024",
                    "mqa_r1_k3_h16_s1_d64_c1024",
                    "uir_r1_a0_k0_s1_e2_c1024",
                    "mqa_r1_k3_h16_s1_d64_c1024",
                    "uir_r1_a0_k0_s1_e2_c1024",
                    "mqa_r1_k3_h16_s1_d64_c1024",
                    "uir_r1_a0_k0_s1_e2_c1024",
                    "mqa_r1_k3_h16_s1_d64_c1024",
                    "uir_r1_a0_k0_s1_e2_c1024",
                    "mqa_r1_k3_h16_s1_d64_c1024",
                    "uir_r1_a0_k0_s1_e2_c1024",
                    "mqa_r1_k3_h16_s1_d64_c1024",
                    "uir_r1_a0_k0_s1_e2_c1024",
                    "mqa_r1_k3_h16_s1_d64_c1024",
                    "uir_r1_a0_k0_s1_e2_c1024",
                ],
            ],
            "image_shape": (256, 256, 3),
            "stem_size": 64,
            "num_features": 2048,
            "msfa_indices": [-3, -2, -1],
            "msfa_output_resolution": 16,
        },
        "kaggle_handle": None,  # No pretrained weights
    },
}

backbone_presets = {
    **backbone_presets_base,
    # Add other presets here if available
}
