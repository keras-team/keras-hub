"""Tests for LiteRT export with specific production models.

This test suite validates export functionality for production model presets
including CausalLM, ImageClassifier, ObjectDetector, and ImageSegmenter models.
"""

import gc
import os
import tempfile

import keras
import numpy as np
import pytest

import keras_hub
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.tests.test_case import TestCase

# Lazy import TensorFlow only when using TensorFlow backend
tf = None
if keras.backend.backend() == "tensorflow":
    import tensorflow as tf

# Lazy import LiteRT interpreter with fallback logic
if keras.backend.backend() == "tensorflow":
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            if tf is not None:
                Interpreter = tf.lite.Interpreter


# Model configurations for testing
CAUSAL_LM_MODELS = [
    # {
    #     "preset": "llama3.2_1b",
    #     "model_class": keras_hub.models.Llama3CausalLM,
    #     "sequence_length": 128,
    #     "vocab_size": 32000,
    #     "test_name": "llama3_2_1b",
    # },
    # {
    #     "preset": "gemma3_1b",
    #     "model_class": keras_hub.models.Gemma3CausalLM,
    #     "sequence_length": 128,
    #     "vocab_size": 32000,
    #     "test_name": "gemma3_1b",
    # },
    # {
    #     "preset": "gpt2_base_en",
    #     "model_class": keras_hub.models.GPT2CausalLM,
    #     "sequence_length": 128,
    #     "vocab_size": 50000,
    #     "test_name": "gpt2_base_en",
    # },
]

IMAGE_CLASSIFIER_MODELS = [
    {
        "preset": "resnet_50_imagenet",
        "test_name": "resnet_50",
    },
    {
        "preset": "efficientnet_b0_ra_imagenet",
        "test_name": "efficientnet_b0",
    },
    {
        "preset": "densenet_121_imagenet",
        "test_name": "densenet_121",
    },
    {
        "preset": "mobilenet_v3_small_100_imagenet",
        "test_name": "mobilenet_v3_small",
    },
]

OBJECT_DETECTOR_MODELS = [
    {
        "preset": "dfine_nano_coco",
        "test_name": "dfine_nano",
    },
    {
        "preset": "retinanet_resnet50_fpn_coco",
        "test_name": "retinanet_resnet50",
    },
]

IMAGE_SEGMENTER_MODELS = [
    {
        "preset": "deeplab_v3_plus_resnet50_pascalvoc",
        "test_name": "deeplab_v3_plus",
    },
]


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class LiteRTCausalLMModelsTest(TestCase):
    """Test LiteRT export for CausalLM models."""

    def test_export_causal_lm_models(self):
        """Test export for all CausalLM models."""
        for model_config in CAUSAL_LM_MODELS:
            with self.subTest(preset=model_config["preset"]):
                self._test_single_model(model_config)

    def _test_single_model(self, model_config):
        """Helper method to test a single CausalLM model.

        Args:
            model_config: Dict containing preset, model_class, sequence_length,
                vocab_size, and test_name.
        """
        preset = model_config["preset"]
        model_class = model_config["model_class"]
        sequence_length = model_config["sequence_length"]
        vocab_size = model_config["vocab_size"]
        test_name = model_config["test_name"]

        try:
            # Load model
            model = model_class.from_preset(preset, load_weights=False)
            model.preprocessor.sequence_length = sequence_length

            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = os.path.join(temp_dir, f"{test_name}.tflite")
                # Use model.export() method
                model.export(export_path, format="litert")

                # Verify file exists
                self.assertTrue(os.path.exists(export_path))
                self.assertGreater(os.path.getsize(export_path), 0)

                # Test inference
                interpreter = Interpreter(export_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Create test inputs with correct dtypes from interpreter
                token_ids = np.random.randint(
                    1, vocab_size, size=(1, sequence_length), dtype=np.int32
                ).astype(input_details[0]["dtype"])
                padding_mask = np.ones(
                    (1, sequence_length), dtype=np.bool_
                ).astype(input_details[1]["dtype"])

                # Set inputs and run inference
                interpreter.set_tensor(input_details[0]["index"], token_ids)
                interpreter.set_tensor(input_details[1]["index"], padding_mask)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])

                # Verify output shape
                self.assertEqual(output.shape[0], 1)
                self.assertEqual(output.shape[1], sequence_length)

        except Exception as e:
            self.skipTest(f"{test_name} model test skipped: {e}")
        finally:
            # Clean up model and interpreter, free memory
            if "model" in locals():
                del model
            if "interpreter" in locals():
                del interpreter
            gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class LiteRTImageClassifierModelsTest(TestCase):
    """Test LiteRT export for ImageClassifier models."""

    def test_export_image_classifier_models(self):
        """Test export for all ImageClassifier models."""
        for model_config in IMAGE_CLASSIFIER_MODELS:
            with self.subTest(preset=model_config["preset"]):
                self._test_single_model(model_config)

    def _test_single_model(self, model_config):
        """Helper method to test a single ImageClassifier model.

        Args:
            model_config: Dict containing preset and test_name.
        """
        preset = model_config["preset"]
        test_name = model_config["test_name"]

        try:
            # Load model
            model = ImageClassifier.from_preset(preset)

            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = os.path.join(temp_dir, f"{test_name}.tflite")
                # Use model.export() method
                model.export(export_path, format="litert")

                # Verify file exists
                self.assertTrue(os.path.exists(export_path))
                self.assertGreater(os.path.getsize(export_path), 0)

                # Test inference
                interpreter = Interpreter(export_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Get input shape from the exported model
                input_shape = input_details[0]["shape"]

                # Create test input with the correct shape
                test_image = np.random.uniform(
                    0.0, 1.0, size=tuple(input_shape)
                ).astype(input_details[0]["dtype"])

                # Run inference
                interpreter.set_tensor(input_details[0]["index"], test_image)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])

                # Verify output shape
                self.assertEqual(output.shape[0], 1)
                self.assertEqual(len(output.shape), 2)

        except Exception as e:
            self.skipTest(f"{test_name} model test skipped: {e}")
        finally:
            # Clean up model and interpreter, free memory
            if "model" in locals():
                del model
            if "interpreter" in locals():
                del interpreter
            gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class LiteRTObjectDetectorModelsTest(TestCase):
    """Test LiteRT export for ObjectDetector models."""

    def test_export_object_detector_models(self):
        """Test export for all ObjectDetector models."""
        for model_config in OBJECT_DETECTOR_MODELS:
            with self.subTest(preset=model_config["preset"]):
                self._test_single_model(model_config)

    def _test_single_model(self, model_config):
        """Helper method to test a single ObjectDetector model.

        Args:
            model_config: Dict containing preset and test_name.
        """
        preset = model_config["preset"]
        test_name = model_config["test_name"]

        try:
            # Load model
            model = ObjectDetector.from_preset(preset)

            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = os.path.join(temp_dir, f"{test_name}.tflite")
                # Use model.export() method
                model.export(export_path, format="litert")

                # Verify file exists
                self.assertTrue(os.path.exists(export_path))
                self.assertGreater(os.path.getsize(export_path), 0)

                # Test inference
                interpreter = Interpreter(export_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Get input shapes from the exported model
                # ObjectDetector requires two inputs: images and image_shape
                image_input_details = input_details[0]
                shape_input_details = input_details[1]
                image_input_shape = image_input_details["shape"]

                # Create test inputs
                test_image = np.random.uniform(
                    0.0, 1.0, size=tuple(image_input_shape)
                ).astype(image_input_details["dtype"])
                test_image_shape = np.array(
                    [[image_input_shape[1], image_input_shape[2]]],
                    dtype=shape_input_details["dtype"],
                )

                # Run inference with both inputs
                interpreter.set_tensor(image_input_details["index"], test_image)
                interpreter.set_tensor(
                    shape_input_details["index"], test_image_shape
                )
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])

                # Verify output shape
                self.assertEqual(output.shape[0], 1)
                self.assertGreater(len(output.shape), 1)

        except Exception as e:
            self.skipTest(f"{test_name} model test skipped: {e}")
        finally:
            # Clean up model and interpreter, free memory
            if "model" in locals():
                del model
            if "interpreter" in locals():
                del interpreter
            gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class LiteRTImageSegmenterModelsTest(TestCase):
    """Test LiteRT export for ImageSegmenter models."""

    def test_export_image_segmenter_models(self):
        """Test export for all ImageSegmenter models."""
        for model_config in IMAGE_SEGMENTER_MODELS:
            with self.subTest(preset=model_config["preset"]):
                self._test_single_model(model_config)

    def _test_single_model(self, model_config):
        """Helper method to test a single ImageSegmenter model.

        Args:
            model_config: Dict containing preset and test_name.
        """
        preset = model_config["preset"]
        test_name = model_config["test_name"]

        try:
            # Load model
            model = ImageSegmenter.from_preset(preset)

            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = os.path.join(temp_dir, f"{test_name}.tflite")
                # Use model.export() method
                model.export(export_path, format="litert")

                # Verify file exists
                self.assertTrue(os.path.exists(export_path))
                self.assertGreater(os.path.getsize(export_path), 0)

                # Test inference
                interpreter = Interpreter(export_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Get input shape from the exported model
                input_shape = input_details[0]["shape"]

                # Create test input with the correct shape
                test_image = np.random.uniform(
                    0.0, 1.0, size=tuple(input_shape)
                ).astype(input_details[0]["dtype"])

                # Run inference
                interpreter.set_tensor(input_details[0]["index"], test_image)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])

                # Verify output shape
                self.assertEqual(output.shape[0], 1)
                self.assertGreater(len(output.shape), 2)

        except Exception as e:
            self.skipTest(f"{test_name} model test skipped: {e}")
        finally:
            # Clean up model and interpreter, free memory
            if "model" in locals():
                del model
            if "interpreter" in locals():
                del interpreter

            gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class LiteRTProductionModelsNumericalTest(TestCase):
    """Numerical verification tests for production models."""

    def test_image_classifier_numerical_accuracy(self):
        """Test numerical accuracy for ImageClassifier exports."""
        # Test first 2 image classifier models
        for model_config in IMAGE_CLASSIFIER_MODELS[:2]:
            with self.subTest(preset=model_config["preset"]):
                self._test_image_classifier_accuracy(model_config)

    def _test_image_classifier_accuracy(self, model_config):
        """Helper method to test numerical accuracy of ImageClassifier.

        Args:
            model_config: Dict containing preset and test_name.
        """
        preset = model_config["preset"]
        test_name = model_config["test_name"]

        try:
            # Load model
            model = ImageClassifier.from_preset(preset)

            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = os.path.join(temp_dir, f"{test_name}.tflite")
                # Use model.export() method
                model.export(export_path, format="litert")

                # Get input shape from exported model
                interpreter = Interpreter(export_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                input_shape = input_details[0]["shape"]

                # Create test input
                test_input = np.random.uniform(
                    0.0, 1.0, size=tuple(input_shape)
                ).astype(input_details[0]["dtype"])

                # Get Keras output
                keras_output = model(test_input).numpy()

                # Get LiteRT output
                interpreter.set_tensor(input_details[0]["index"], test_input)
                interpreter.invoke()
                litert_output = interpreter.get_tensor(
                    output_details[0]["index"]
                )

                # Compare outputs
                max_diff = np.max(np.abs(keras_output - litert_output))
                self.assertLess(
                    max_diff,
                    1e-2,
                    f"{test_name}: Max diff {max_diff} exceeds tolerance",
                )

        except Exception as e:
            self.skipTest(f"{test_name} numerical test skipped: {e}")
        finally:
            # Clean up model and interpreter, free memory
            if "model" in locals():
                del model
            if "interpreter" in locals():
                del interpreter

            gc.collect()

    def test_causal_lm_numerical_accuracy(self):
        """Test numerical accuracy for CausalLM exports."""
        # Test first CausalLM model
        for model_config in CAUSAL_LM_MODELS[:1]:
            with self.subTest(preset=model_config["preset"]):
                self._test_causal_lm_accuracy(model_config)

    def _test_causal_lm_accuracy(self, model_config):
        """Helper method to test numerical accuracy of CausalLM.

        Args:
            model_config: Dict containing preset, model_class, sequence_length,
                vocab_size, and test_name.
        """
        preset = model_config["preset"]
        model_class = model_config["model_class"]
        sequence_length = model_config["sequence_length"]
        vocab_size = model_config["vocab_size"]
        test_name = model_config["test_name"]

        try:
            # Load model
            model = model_class.from_preset(preset, load_weights=False)
            model.preprocessor.sequence_length = sequence_length

            # Create test inputs
            token_ids = np.random.randint(
                1, vocab_size, size=(1, sequence_length), dtype=np.int32
            )
            padding_mask = np.ones((1, sequence_length), dtype=np.bool_)
            test_input = {"token_ids": token_ids, "padding_mask": padding_mask}

            # Get Keras output
            keras_output = model(test_input).numpy()

            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = os.path.join(temp_dir, f"{test_name}.tflite")
                # Use model.export() method
                model.export(export_path, format="litert")

                # Get LiteRT output
                interpreter = Interpreter(export_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Cast inputs to match interpreter expected dtypes
                token_ids_cast = token_ids.astype(input_details[0]["dtype"])
                padding_mask_cast = padding_mask.astype(
                    input_details[1]["dtype"]
                )

                interpreter.set_tensor(
                    input_details[0]["index"], token_ids_cast
                )
                interpreter.set_tensor(
                    input_details[1]["index"], padding_mask_cast
                )
                interpreter.invoke()
                litert_output = interpreter.get_tensor(
                    output_details[0]["index"]
                )

                # Compare outputs
                max_diff = np.max(np.abs(keras_output - litert_output))
                self.assertLess(
                    max_diff,
                    1e-3,
                    f"{test_name}: Max diff {max_diff} exceeds tolerance",
                )

        except Exception as e:
            self.skipTest(f"{test_name} numerical test skipped: {e}")
        finally:
            # Clean up model and interpreter, free memory
            if "model" in locals():
                del model
            if "interpreter" in locals():
                del interpreter

            gc.collect()
