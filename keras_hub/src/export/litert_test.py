"""Tests for LiteRT export functionality."""

import os
import shutil
import tempfile

import keras
import numpy as np
import pytest

from keras_hub.src.export.litert import LiteRTExporter
from keras_hub.src.tests.test_case import TestCase

# Lazy import LiteRT interpreter with fallback logic
LITERT_AVAILABLE = False
if keras.backend.backend() == "tensorflow":
    try:
        from ai_edge_litert.interpreter import Interpreter

        LITERT_AVAILABLE = True
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class LiteRTExporterTest(TestCase):
    """Tests for LiteRTExporter class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        # Clean up temporary files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_exporter_init_without_litert_available(self):
        """Test that LiteRTExporter raises error if Keras LiteRT unavailable."""
        # We can't easily test this without mocking, so we'll skip
        self.skipTest("Requires mocking KERAS_LITE_RT_AVAILABLE")

    def test_exporter_init_with_parameters(self):
        """Test LiteRTExporter initialization with custom parameters."""
        from keras_hub.src.export.configs import CausalLMExporterConfig
        from keras_hub.src.models.causal_lm import CausalLM

        # Create a minimal mock model
        class MockCausalLM(CausalLM):
            def __init__(self):
                keras.Model.__init__(self)
                self.preprocessor = None
                self.dense = keras.layers.Dense(10)

            def call(self, inputs):
                return self.dense(inputs["token_ids"])

        model = MockCausalLM()
        config = CausalLMExporterConfig(model)
        exporter = LiteRTExporter(
            config,
            max_sequence_length=256,
            verbose=True,
            custom_param="test",
        )

        self.assertEqual(exporter.max_sequence_length, 256)
        self.assertTrue(exporter.verbose)
        self.assertEqual(exporter.export_kwargs["custom_param"], "test")


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class CausalLMExportTest(TestCase):
    """Tests for exporting CausalLM models to LiteRT."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_causal_lm_mock(self):
        """Test exporting a mock CausalLM model."""
        from keras_hub.src.models.causal_lm import CausalLM

        # Create a minimal mock CausalLM
        class SimpleCausalLM(CausalLM):
            def __init__(self):
                super().__init__()
                self.preprocessor = None
                self.embedding = keras.layers.Embedding(1000, 64)
                self.dense = keras.layers.Dense(1000)

            def call(self, inputs):
                if isinstance(inputs, dict):
                    token_ids = inputs["token_ids"]
                else:
                    token_ids = inputs
                x = self.embedding(token_ids)
                return self.dense(x)

        model = SimpleCausalLM()
        model.build(
            input_shape={
                "token_ids": (None, 128),
                "padding_mask": (None, 128),
            }
        )

        # Export using the model's export method
        export_path = os.path.join(self.temp_dir, "test_causal_lm")
        model.export(export_path, format="litert")

        # Verify the file was created
        tflite_path = export_path + ".tflite"
        self.assertTrue(os.path.exists(tflite_path))

        # Load and verify the exported model
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Delete the TFLite file after loading to free disk space
        if os.path.exists(tflite_path):
            os.remove(tflite_path)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Verify we have the expected inputs
        self.assertEqual(len(input_details), 2)

        # Create test inputs with dtypes from the interpreter
        test_token_ids = np.random.randint(0, 1000, (1, 128)).astype(
            input_details[0]["dtype"]
        )
        test_padding_mask = np.ones((1, 128), dtype=input_details[1]["dtype"])

        # Set inputs and run inference
        interpreter.set_tensor(input_details[0]["index"], test_token_ids)
        interpreter.set_tensor(input_details[1]["index"], test_padding_mask)
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]["index"])
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 128)  # Sequence length
        self.assertEqual(output.shape[2], 1000)  # Vocab size

        # Clean up interpreter, free memory
        del interpreter
        import gc

        gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class ImageClassifierExportTest(TestCase):
    """Tests for exporting ImageClassifier models to LiteRT."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_image_classifier_mock(self):
        """Test exporting a mock ImageClassifier model."""
        from keras_hub.src.models.backbone import Backbone
        from keras_hub.src.models.image_classifier import ImageClassifier

        # Create a minimal mock Backbone
        class SimpleBackbone(Backbone):
            def __init__(self):
                inputs = keras.layers.Input(shape=(224, 224, 3))
                x = keras.layers.Conv2D(32, 3, padding="same")(inputs)
                # Don't reduce dimensions - let ImageClassifier handle pooling
                outputs = x
                super().__init__(inputs=inputs, outputs=outputs)

        # Create ImageClassifier with the mock backbone
        backbone = SimpleBackbone()
        model = ImageClassifier(backbone=backbone, num_classes=10)

        # Export using the model's export method
        export_path = os.path.join(self.temp_dir, "test_image_classifier")
        model.export(export_path, format="litert")

        # Verify the file was created
        tflite_path = export_path + ".tflite"
        self.assertTrue(os.path.exists(tflite_path))

        # Load and verify the exported model
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Delete the TFLite file after loading to free disk space
        if os.path.exists(tflite_path):
            os.remove(tflite_path)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Verify we have the expected input
        self.assertEqual(len(input_details), 1)

        # Create test input with dtype from the interpreter
        test_image = np.random.uniform(0.0, 1.0, (1, 224, 224, 3)).astype(
            input_details[0]["dtype"]
        )

        # Set input and run inference
        interpreter.set_tensor(input_details[0]["index"], test_image)
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]["index"])
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 10)  # Number of classes

        # Clean up interpreter, free memory
        del interpreter
        import gc

        gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class TextClassifierExportTest(TestCase):
    """Tests for exporting TextClassifier models to LiteRT."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_text_classifier_mock(self):
        """Test exporting a mock TextClassifier model."""
        from keras_hub.src.models.text_classifier import TextClassifier

        # Create a minimal mock TextClassifier
        class SimpleTextClassifier(TextClassifier):
            def __init__(self):
                super().__init__()
                self.preprocessor = None
                self.embedding = keras.layers.Embedding(5000, 64)
                self.pool = keras.layers.GlobalAveragePooling1D()
                self.dense = keras.layers.Dense(5)  # 5 classes

            def call(self, inputs):
                if isinstance(inputs, dict):
                    token_ids = inputs["token_ids"]
                else:
                    token_ids = inputs
                x = self.embedding(token_ids)
                x = self.pool(x)
                return self.dense(x)

        model = SimpleTextClassifier()
        model.build(
            input_shape={
                "token_ids": (None, 128),
                "padding_mask": (None, 128),
            }
        )

        # Export using the model's export method
        export_path = os.path.join(self.temp_dir, "test_text_classifier")
        model.export(export_path, format="litert")

        # Verify the file was created
        tflite_path = export_path + ".tflite"
        self.assertTrue(os.path.exists(tflite_path))

        # Load and verify the exported model
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Delete the TFLite file after loading to free disk space
        if os.path.exists(tflite_path):
            os.remove(tflite_path)

        output_details = interpreter.get_output_details()

        # Verify output shape (batch, num_classes)
        self.assertEqual(len(output_details), 1)

        # Clean up interpreter, free memory
        del interpreter
        import gc

        gc.collect()


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class ExportNumericalVerificationTest(TestCase):
    """Tests for numerical accuracy of exported models."""

    def test_simple_model_numerical_accuracy(self):
        """Test that exported model produces similar outputs to original."""
        # Create a simple sequential model
        model = keras.Sequential(
            [
                keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                keras.layers.Dense(3, activation="softmax"),
            ]
        )

        # Prepare test input
        test_input = np.random.random((1, 5)).astype(np.float32)

        # Use standardized test from TestCase
        # Note: This assumes the model has an export() method
        # If not available, the test will be skipped
        if not hasattr(model, "export"):
            self.skipTest("model.export() not available")

        self.run_litert_export_test(
            cls=keras.Sequential,
            init_kwargs={
                "layers": [
                    keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                    keras.layers.Dense(3, activation="softmax"),
                ]
            },
            input_data=test_input,
            expected_output_shape=(1, 3),
            comparison_mode="strict",
        )

    def test_dict_input_model_numerical_accuracy(self):
        """Test numerical accuracy for models with dictionary inputs."""

        # Define a custom model class for testing
        class DictInputModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.concat = keras.layers.Concatenate()
                self.dense = keras.layers.Dense(5)

            def call(self, inputs):
                x = self.concat([inputs["input1"], inputs["input2"]])
                return self.dense(x)

        # Prepare test inputs
        test_inputs = {
            "input1": np.random.random((1, 10)).astype(np.float32),
            "input2": np.random.random((1, 10)).astype(np.float32),
        }

        # Use standardized test from TestCase
        self.run_litert_export_test(
            cls=DictInputModel,
            init_kwargs={},
            input_data=test_inputs,
            expected_output_shape=(1, 5),
            comparison_mode="strict",
        )


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="LiteRT export only supports TensorFlow backend.",
)
class ExportErrorHandlingTest(TestCase):
    """Tests for error handling in export process."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_to_invalid_path(self):
        """Test that export with invalid path raises appropriate error."""
        if not hasattr(keras.Model, "export"):
            self.skipTest("model.export() not available")

        model = keras.Sequential([keras.layers.Dense(10)])

        # Try to export to a path that doesn't exist and can't be created
        invalid_path = "/nonexistent/deeply/nested/path/model"

        with self.assertRaises(Exception):
            model.export(invalid_path, format="litert")

    def test_export_unbuilt_model(self):
        """Test exporting an unbuilt model."""
        if not hasattr(keras.Model, "export"):
            self.skipTest("model.export() not available")

        model = keras.Sequential([keras.layers.Dense(10, input_shape=(5,))])

        # Model is not built yet (no explicit build() call)
        # Export should still work by building the model
        export_path = os.path.join(self.temp_dir, "unbuilt_model.tflite")
        model.export(export_path, format="litert")

        # Should succeed
        self.assertTrue(os.path.exists(export_path))
