"""Convert Xception model to Keras Hub."

Keras applications only has a single Xception model, so we will just convert
that and take no flags here.
"""

import keras
import numpy as np

import keras_hub


def convert(applications_model):
    backbone = keras_hub.models.XceptionBackbone(
        stackwise_conv_filters=(
            [[32, 64], [128, 128], [256, 256], [728, 728]]
            + [[728, 728, 728]] * 8
            + [[728, 1024], [1536, 2048]]
        ),
        stackwise_pooling=(
            [False, True, True, True] + [False] * 8 + [True, False]
        ),
    )
    image_converter = keras_hub.layers.ImageConverter(
        image_size=(299, 299),
        scale=1.0 / 127.5,
        offset=-1.0,
    )
    preprocessor = keras_hub.models.XceptionImageClassifierPreprocessor(
        image_converter=image_converter,
    )
    hub_classifier = keras_hub.models.XceptionImageClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=1000,
    )
    assert hub_classifier.count_params() == applications_model.count_params()
    # For now, assume a one-to-one iteration order of weights.
    # We could make this more robust in the future.
    for hub_weight, model_weight in zip(
        hub_classifier.weights, applications_model.weights
    ):
        hub_weight.assign(model_weight.value)
    return hub_classifier


def validate(hub_model, applications_model):
    image = np.random.randint(0, 255, (1, 299, 299, 3)).astype("float32")
    scaled_image = hub_model.preprocessor(image)
    ref_scaled_image = keras.applications.xception.preprocess_input(image)
    np.testing.assert_allclose(scaled_image, ref_scaled_image, atol=1e-5)
    no_top = keras.applications.Xception(weights="imagenet", include_top=False)
    ref_backbone_output = no_top(scaled_image)
    backbone_output = hub_model.backbone(scaled_image)
    np.testing.assert_allclose(backbone_output, ref_backbone_output)
    # Keras applications uses a softmax by default.
    preds = keras.ops.softmax(hub_model(scaled_image))
    ref_preds = applications_model(scaled_image)
    np.testing.assert_allclose(preds, ref_preds)


if __name__ == "__main__":
    print("✅ keras.applicaitons model loaded")
    applications_model = keras.applications.Xception(weights="imagenet")
    print("✅ KerasHub model converted")
    hub_model = convert(applications_model)
    print("✅ Weights converted")
    validate(hub_model, applications_model)
    print("✅ Output validated")
    preset = "xception_41_imagenet"
    hub_model.save_to_preset("xception_41_imagenet")
    print(f"🏁 Preset saved to ./{preset}")
