from pathlib import Path

import numpy as np
from keras import ops
from keras_cv.models import YOLOV8Backbone as KerasCVYOLOV8Backbone
from keras_hub.models import YOLOV8Backbone


def validate_numerics(rng, preset_name, model_A, model_B):
    random_data = rng.random((2, 224, 224, 3))
    output_A = model_A(random_data)
    output_B = model_B(random_data)
    output_A = ops.convert_to_numpy(output_A)
    output_B = ops.convert_to_numpy(output_B)
    is_valid = np.allclose(output_A, output_B)
    print(f"Port {preset_name} with valid numerics: {is_valid}")
    assert is_valid


def make_directory(root, preset):
    preset_path_name = f"{root}/{preset}"
    Path(preset_path_name).mkdir(parents=True, exist_ok=True)
    return preset_path_name


def pass_weights_A_to_B(model_A, model_B, root_path):
    weights_filepath = f"{root_path}.weights.h5"
    model_A.save_weights(weights_filepath)
    model_B.load_weights(weights_filepath)
    return model_A, model_B


def convert_backbone(ModelA, ModelB, weights_path, preset_name):
    preset_path = make_directory(weights_path, preset_name)
    model_A = ModelA.from_preset(preset_name)
    model_B = ModelB(**model_A.get_config())
    model_A, model_B = pass_weights_A_to_B(model_A, model_B, preset_path)
    model_B.save_to_preset(preset_path)
    return model_A, model_B


if __name__ == "__main__":
    import argparse
    from functools import partial
    description = "Convert YOLOV8 keras-cv to keras-hub weights."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--weights_path", type=str, default="YOLOV8")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    backbone_presets = [
        "yolo_v8_xs_backbone",
        "yolo_v8_s_backbone",
        "yolo_v8_m_backbone",
        "yolo_v8_l_backbone",
        "yolo_v8_xl_backbone",
        "yolo_v8_xs_backbone_coco",
        "yolo_v8_s_backbone_coco",
        "yolo_v8_m_backbone_coco",
        "yolo_v8_l_backbone_coco",
        "yolo_v8_xl_backbone_coco"
    ]

    convert = partial(convert_backbone, KerasCVYOLOV8Backbone, YOLOV8Backbone)
    for preset in backbone_presets:
        model_A, model_B = convert(args.weights_path, preset)
        validate_numerics(rng, preset, model_A, model_B)
