from .controlnet import ControlNet


controlnet_presets = {
    "controlnet_base": {
        "description": "Minimal ControlNet base configuration.",
        "config": {
            "image_size": 128,
            "base_channels": 64,
        },
    }
}


def from_preset(preset_name):
    if preset_name not in controlnet_presets:
        raise ValueError(f"Unknown preset: {preset_name}")

    config = controlnet_presets[preset_name]["config"]
    return ControlNet(**config)
