"""Differentiable Binarization preset configurations."""

backbone_presets = {
    "diffbin_r50vd_icdar2015": {
        "metadata": {
            "description": (
                "Differentiable Binarization using 50-layer"
                "ResNetVD trained on the ICDAR2015 dataset."
            ),
            "params": 25482722,
            "official_name": "DifferentiableBinarization",
            "path": "diffbin",
            "model_card": "https://arxiv.org/abs/1911.08947",
        },
        "kaggle_handle": "",  # TODO
    }
}