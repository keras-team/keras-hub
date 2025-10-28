"""Tests for LiteRT export with specific production models.

This test suite validates LiteRT export functionality for production
model presets including CausalLM, ImageClassifier, ObjectDetector,
and ImageSegmenter models.

Each test validates export correctness by:
1. Loading a model from preset
2. Exporting it to LiteRT format
3. Running numerical verification to ensure exported model produces
   equivalent outputs
4. Comparing outputs statistically against predefined thresholds

This ensures that exported models maintain functional correctness and
numerical stability.
"""

import gc

import keras
import numpy as np
import pytest

from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.tests.test_case import TestCase

# Model configurations for testing
CAUSAL_LM_MODELS = [
    {
        "preset": "llama3.2_1b",
        "model_class": Llama3CausalLM,
        "sequence_length": 128,
        "test_name": "llama3_2_1b",
        "output_thresholds": {"*": {"max": 1e-3, "mean": 1e-5}},
    },
    {
        "preset": "gemma3_1b",
        "model_class": Gemma3CausalLM,
        "sequence_length": 128,
        "test_name": "gemma3_1b",
        "output_thresholds": {"*": {"max": 1e-3, "mean": 3e-5}},
    },
    {
        "preset": "gpt2_base_en",
        "model_class": GPT2CausalLM,
        "sequence_length": 128,
        "test_name": "gpt2_base_en",
        "output_thresholds": {"*": {"max": 5e-4, "mean": 5e-5}},
    },
]

IMAGE_CLASSIFIER_MODELS = [
    {
        "preset": "resnet_50_imagenet",
        "test_name": "resnet_50",
        "input_range": (0.0, 1.0),
        "output_thresholds": {"*": {"max": 5e-5, "mean": 1e-5}},
    },
    {
        "preset": "efficientnet_b0_ra_imagenet",
        "test_name": "efficientnet_b0",
        "input_range": (0.0, 1.0),
        "output_thresholds": {"*": {"max": 5e-5, "mean": 1e-5}},
    },
    {
        "preset": "densenet_121_imagenet",
        "test_name": "densenet_121",
        "input_range": (0.0, 1.0),
        "output_thresholds": {"*": {"max": 5e-5, "mean": 1e-5}},
    },
    {
        "preset": "mobilenet_v3_small_100_imagenet",
        "test_name": "mobilenet_v3_small",
        "input_range": (0.0, 1.0),
        "output_thresholds": {"*": {"max": 5e-5, "mean": 1e-5}},
    },
]

OBJECT_DETECTOR_MODELS = [
    {
        "preset": "dfine_small_coco",
        "test_name": "dfine_small",
        "input_range": (0.0, 1.0),
        "output_thresholds": {
            "intermediate_predicted_corners": {"max": 5.0, "mean": 0.05},
            "intermediate_logits": {"max": 5.0, "mean": 0.1},
            "enc_topk_logits": {"max": 5.0, "mean": 0.03},
            "logits": {"max": 2.0, "mean": 0.03},
            "*": {"max": 1.0, "mean": 0.03},
        },
    },
    {
        "preset": "dfine_medium_coco",
        "test_name": "dfine_medium",
        "input_range": (0.0, 1.0),
        "output_thresholds": {
            "intermediate_predicted_corners": {"max": 50.0, "mean": 0.15},
            "intermediate_logits": {"max": 5.0, "mean": 0.1},
            "enc_topk_logits": {"max": 5.0, "mean": 0.03},
            "logits": {"max": 2.0, "mean": 0.03},
            "*": {"max": 1.0, "mean": 0.03},
        },
    },
    {
        "preset": "retinanet_resnet50_fpn_coco",
        "test_name": "retinanet_resnet50",
        "input_range": (0.0, 1.0),
        "output_thresholds": {
            "enc_topk_logits": {"max": 5.0, "mean": 0.03},
            "logits": {"max": 2.0, "mean": 0.03},
            "*": {"max": 1.0, "mean": 0.03},
        },
    },
]

IMAGE_SEGMENTER_MODELS = [
    {
        "preset": "deeplab_v3_plus_resnet50_pascalvoc",
        "test_name": "deeplab_v3_plus",
        "input_range": (0.0, 1.0),
        "output_thresholds": {"*": {"max": 1.0, "mean": 1e-2}},
    },
]


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
@pytest.mark.parametrize(
    "model_config",
    CAUSAL_LM_MODELS,
    ids=lambda x: f"{x['test_name']}-{x['preset']}",
)
def test_causal_lm_litert_export(model_config):
    """Test LiteRT export for CausalLM models.

    Validates that the model can be successfully exported to LiteRT format
    and produces numerically equivalent outputs.
    """
    preset = model_config["preset"]
    model_class = model_config["model_class"]
    sequence_length = model_config["sequence_length"]
    output_thresholds = model_config.get(
        "output_thresholds", {"*": {"max": 3e-5, "mean": 3e-6}}
    )

    model = None
    try:
        # Load model from preset once
        model = model_class.from_preset(preset, load_weights=True)

        # Set sequence length before export
        model.preprocessor.sequence_length = sequence_length

        # Get vocab_size from the loaded model
        vocab_size = model.backbone.vocabulary_size

        # Prepare test inputs with fixed random seed for reproducibility
        np.random.seed(42)
        input_data = {
            "token_ids": np.random.randint(
                1, vocab_size, size=(1, sequence_length), dtype=np.int32
            ),
            "padding_mask": np.ones((1, sequence_length), dtype=np.int32),
        }

        # Validate LiteRT export with numerical verification
        TestCase().run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=(1, sequence_length, vocab_size),
            comparison_mode="statistical",
            output_thresholds=output_thresholds,
        )

    finally:
        # Clean up model, free memory
        if model is not None:
            del model
        gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
@pytest.mark.parametrize(
    "model_config",
    IMAGE_CLASSIFIER_MODELS,
    ids=lambda x: f"{x['test_name']}-{x['preset']}",
)
def test_image_classifier_litert_export(model_config):
    """Test LiteRT export for ImageClassifier models.

    Validates that the model can be successfully exported to LiteRT format
    and produces numerically equivalent outputs.
    """
    preset = model_config["preset"]
    input_range = model_config.get("input_range", (0.0, 1.0))
    output_thresholds = model_config.get(
        "output_thresholds", {"*": {"max": 1e-4, "mean": 4e-5}}
    )

    model = None
    try:
        # Load model once
        model = ImageClassifier.from_preset(preset)

        # Get actual image size from model preprocessor or backbone
        image_size = getattr(model.preprocessor, "image_size", None)
        if image_size is None and hasattr(model.backbone, "image_shape"):
            image_shape = model.backbone.image_shape
            if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
                image_size = tuple(image_shape[:2])
            elif isinstance(image_shape, int):
                image_size = (image_shape, image_shape)

        if image_size is None:
            raise ValueError(f"Could not determine image size for {preset}")

        input_shape = image_size + (3,)  # Add channels

        # Prepare test input
        test_image = np.random.uniform(
            input_range[0], input_range[1], size=(1,) + input_shape
        ).astype(np.float32)

        # Validate LiteRT export with numerical verification
        TestCase().run_litert_export_test(
            model=model,
            input_data=test_image,
            expected_output_shape=None,  # Output shape varies by model
            comparison_mode="statistical",
            output_thresholds=output_thresholds,
        )

    finally:
        # Clean up model, free memory
        if model is not None:
            del model
        gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
@pytest.mark.parametrize(
    "model_config",
    OBJECT_DETECTOR_MODELS,
    ids=lambda x: f"{x['test_name']}-{x['preset']}",
)
def test_object_detector_litert_export(model_config):
    """Test LiteRT export for ObjectDetector models.

    Validates that the model can be successfully exported to LiteRT format
    and produces numerically equivalent outputs.
    """
    preset = model_config["preset"]
    input_range = model_config.get("input_range", (0.0, 1.0))
    output_thresholds = model_config.get(
        "output_thresholds", {"*": {"max": 1.0, "mean": 0.02}}
    )

    model = None
    try:
        # Load model once
        model = ObjectDetector.from_preset(preset)

        # Get actual image size from model preprocessor or backbone
        image_size = getattr(model.preprocessor, "image_size", None)
        if image_size is None and hasattr(model.backbone, "image_shape"):
            image_shape = model.backbone.image_shape
            if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
                image_size = tuple(image_shape[:2])
            elif isinstance(image_shape, int):
                image_size = (image_shape, image_shape)

        if image_size is None:
            raise ValueError(f"Could not determine image size for {preset}")

        # ObjectDetector typically needs images (H, W, 3) and image_shape (H, W)
        test_inputs = {
            "images": np.random.uniform(
                input_range[0],
                input_range[1],
                size=(1,) + image_size + (3,),
            ).astype(np.float32),
            "image_shape": np.array([image_size], dtype=np.int32),
        }

        # Validate LiteRT export with numerical verification
        TestCase().run_litert_export_test(
            model=model,
            input_data=test_inputs,
            expected_output_shape=None,  # Output varies by model
            comparison_mode="statistical",
            output_thresholds=output_thresholds,
        )

    finally:
        # Clean up model, free memory
        if model is not None:
            del model
        gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
@pytest.mark.parametrize(
    "model_config",
    IMAGE_SEGMENTER_MODELS,
    ids=lambda x: f"{x['test_name']}-{x['preset']}",
)
def test_image_segmenter_litert_export(model_config):
    """Test LiteRT export for ImageSegmenter models.

    Validates that the model can be successfully exported to LiteRT format
    and produces numerically equivalent outputs.
    """
    preset = model_config["preset"]
    input_range = model_config.get("input_range", (0.0, 1.0))
    output_thresholds = model_config.get(
        "output_thresholds", {"*": {"max": 1.0, "mean": 1e-2}}
    )

    model = None
    try:
        # Load model once
        model = ImageSegmenter.from_preset(preset)

        # Get actual image size from model preprocessor or backbone
        image_size = getattr(model.preprocessor, "image_size", None)
        if image_size is None and hasattr(model.backbone, "image_shape"):
            image_shape = model.backbone.image_shape
            if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
                image_size = tuple(image_shape[:2])
            elif isinstance(image_shape, int):
                image_size = (image_shape, image_shape)

        if image_size is None:
            raise ValueError(f"Could not determine image size for {preset}")

        input_shape = image_size + (3,)  # Add channels

        # Prepare test input
        test_image = np.random.uniform(
            input_range[0], input_range[1], size=(1,) + input_shape
        ).astype(np.float32)

        # Validate LiteRT export with numerical verification
        TestCase().run_litert_export_test(
            model=model,
            input_data=test_image,
            expected_output_shape=None,  # Output shape varies by model
            comparison_mode="statistical",
            output_thresholds=output_thresholds,
        )

    finally:
        # Clean up model, free memory
        if model is not None:
            del model
        gc.collect()
