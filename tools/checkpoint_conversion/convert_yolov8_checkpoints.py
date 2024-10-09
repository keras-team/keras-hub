from pathlib import Path
import numpy as np
from keras import ops
from keras_cv.models import YOLOV8Backbone as KerasCVYOLOV8Backbone
from keras_hub.models import YOLOV8Backbone


def validate_numerics(rng, model_A, model_B):
    random_data = rng.random((2, 224, 224, 3))
    output_A = model_A(random_data)
    output_B = model_B(random_data)
    output_A = ops.convert_to_numpy(output_A)
    output_B = ops.convert_to_numpy(output_B)
    all_close = np.allclose(output_A, output_B)
    assert all_close
    return all_close


def make_directory(root, preset):
    preset_path_name = f"{root}/{preset}"
    Path(preset_path_name).mkdir(parents=True, exist_ok=True)
    return preset_path_name


def port_and_validate(rng, weights_path, preset_name):
    preset_path = make_directory(weights_path, preset_name)

    model_A = KerasCVYOLOV8Backbone.from_preset(preset_name)
    model_B = YOLOV8Backbone(**model_A.get_config())

    weights_name = f"{weights_path}/{preset_name}.weights.h5"
    model_A.save_weights(weights_name)
    model_B.load_weights(weights_name)
    model_B.save_to_preset(preset_path)
    is_valid = validate_numerics(rng, model_A, model_B)
    print(f"Port {preset_name} is {is_valid}")


if __name__ == "__main__":
    import argparse
    description = "Convert YOLOV8 keras-cv to keras-hub weights."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--weights_path", type=str, default="YOLOV8")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    presets = [
        "yolo_v8_xs_backbone_coco",
        "yolo_v8_s_backbone_coco",
        "yolo_v8_m_backbone_coco",
        "yolo_v8_l_backbone_coco",
        "yolo_v8_xl_backbone_coco"
    ]
    for preset in presets:
        port_and_validate(rng, args.weights_path, preset)
