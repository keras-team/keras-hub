"""BASNet model preset configurations."""

basnet_presets = {
    "basnet_duts": {
        "metadata": {
            "description": (
                "BASNet model with a 34-layer ResNet backbone, pre-trained "
                "on the DUTS image dataset at a 288x288 resolution. Model "
                "training was performed by Hamid Ali "
                "(https://github.com/hamidriasat/BASNet)."
            ),
            "params": 108886792,
            "path": "basnet",
        },
        "kaggle_handle": "kaggle://keras/basnet/keras/basnet_duts",
    },
}
