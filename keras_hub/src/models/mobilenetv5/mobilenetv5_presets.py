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
                    "uir_r1_a3_k5_s2_e6_c256",  # block 0
                    # FIX: Define each of the next 4 blocks individually
                    "uir_r1_a5_k0_s1_e4_c256",  # block 1 (5x5 kernel)
                    "uir_r1_a3_k0_s1_e4_c256",  # block 2 (3x3 kernel)
                    "uir_r1_a5_k0_s1_e4_c256",  # block 3 (5x5 kernel)
                    "uir_r1_a3_k0_s1_e4_c256",  # block 4 (3x3 kernel)
                ],
                # Stage 2: Corrected channel counts from 512 to 640
                [
                    "uir_r1_a5_k5_s2_e6_c640",    # block 0
                    "uir_r7_a5_k0_s1_e4_c640",    # blocks 1 through 7
                    # FIX: Change e4 to e1 for this specific block
                    "uir_r1_a0_k0_s1_e1_c640",    # block 8
                    "mqa_r1_k3_h12_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                ],
                # Stage 3: Corrected channel counts from 1024 to 1280
                [
                    "uir_r1_a5_k5_s2_e6_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280", # Note: d96 for key_dim
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                ],
            ],
            "image_shape": (256, 256, 3),
            "stem_size": 64,
            "num_features": 2048,
            "msfa_indices": [-15, -1],
            "msfa_output_resolution": 16,
        },
        "kaggle_handle": None,  # No pretrained weights
    },
}

backbone_presets = {
    **backbone_presets_base,
    # Add other presets here if available
}
