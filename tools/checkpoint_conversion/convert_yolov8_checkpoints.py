from pathlib import Path

import numpy as np
from keras import ops
from keras_cv.models import YOLOV8Backbone as KerasCVYOLOV8Backbone
from keras_cv.models import YOLOV8Detector

from keras_hub.layers import YOLOV8ImageConverter
from keras_hub.models import YOLOV8ObjectDetectorPreprocessor
from keras_hub.models import YOLOV8Backbone
from keras_hub.models import YOLOV8ObjectDetector
from keras_hub.models.yolov8 import LabelEncoder
from keras_hub.models.yolov8 import NonMaxSuppression


def get_max_abs_error(output_A, output_B):
    return ops.max(ops.abs(output_A - output_B))


def validate_numerics(rng, preset_name, model_A, model_B, preprocessor_B):
    random_data = rng.random((2, 224, 224, 3)).astype("float32")
    output_A = model_A(random_data)
    if preprocessor_B is not None:
        output_B = model_B.predict(preprocessor_B(random_data))
    else:
        output_B = model_B.predict(random_data)
    output_A = ops.convert_to_numpy(output_A)
    output_B = ops.convert_to_numpy(output_B)
    is_valid = np.allclose(output_A, output_B, atol=1e-5)
    print(f"Port {preset_name} with valid numerics: {is_valid}")
    print("Max abs error", get_max_abs_error(output_A, output_B))
    assert is_valid


def validate_detector_numerics(rng, preset_name, model_A, model_B):
    random_data = rng.random((2, 224, 224, 3)).astype("float32")
    output_A = model_A.predict(random_data)
    output_B = model_B.predict(random_data)
    for key in output_A.keys():
        x_A = output_A[key]
        x_B = output_B[key]
        x_A = ops.convert_to_numpy(x_A)
        x_B = ops.convert_to_numpy(x_B)
        is_valid = np.allclose(x_A, x_B, atol=1e-5)
        print(f"Port '{preset_name}' '{key}' with valid numerics: {is_valid}")
        print("Max abs error", get_max_abs_error(x_A, x_B))
        assert is_valid


def make_directory(root, preset):
    preset_path_name = f"{root}/{preset}"
    Path(preset_path_name).mkdir(parents=True, exist_ok=True)
    return preset_path_name


def pass_weights_A_to_B(model_A, model_B, root_path):
    model_B.set_weights(model_A.get_weights())


def convert_backbone(ModelA, ModelB, weights_path, preset_name):
    preset_path = make_directory(weights_path, preset_name)
    model_A = ModelA.from_preset(preset_name)
    config = model_A.get_config()
    config.pop("include_rescaling")
    model_B = ModelB(**config)
    pass_weights_A_to_B(model_A, model_B, preset_path)
    model_B.save_to_preset(preset_path)
    return model_A, model_B


def build_detector_parts(config):
    backbone_config = config["backbone"]["config"]
    backbone_config.pop("include_rescaling")
    backbone = YOLOV8Backbone(**backbone_config)
    config["backbone"] = backbone
    label_encoder = LabelEncoder(**config["label_encoder"]["config"])
    config["label_encoder"] = label_encoder
    prediction_decoder = NonMaxSuppression(
        **config["prediction_decoder"]["config"]
    )
    config["prediction_decoder"] = prediction_decoder
    return config


def build_preprocessor():
    image_converter = YOLOV8ImageConverter(scale=1.0 / 255)
    preprocessor = YOLOV8ObjectDetectorPreprocessor(
        image_converter=image_converter)
    return preprocessor


def convert_detector(ModelA, ModelB, weights_path, preset_name):
    model_A = ModelA.from_preset(preset_name)
    config = model_A.get_config()
    config = build_detector_parts(config)
    config["preprocessor"] = build_preprocessor()
    print(config)
    model_B = ModelB(**config)
    preset_path = make_directory(weights_path, preset_name)
    pass_weights_A_to_B(model_A, model_B, preset_path)
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
        "yolo_v8_xl_backbone_coco",
    ]
    convert = partial(convert_backbone, KerasCVYOLOV8Backbone, YOLOV8Backbone)
    for preset in backbone_presets:
        model_A, model_B = convert(args.weights_path, preset)
        validate_numerics(rng, preset, model_A, model_B, lambda x: x / 255.0)

    preset = "yolo_v8_m_pascalvoc"
    model_A, model_B = convert_detector(
        YOLOV8Detector, YOLOV8ObjectDetector, args.weights_path, preset
    )
    validate_detector_numerics(rng, preset, model_A, model_B)
