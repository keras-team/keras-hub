"""Tests for LayoutLMv3 document classifier preprocessor."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from ..layoutlmv3.layoutlmv3_document_classifier_preprocessor import LayoutLMv3DocumentClassifierPreprocessor

class LayoutLMv3DocumentClassifierPreprocessorTest(tf.test.TestCase):
    def setUp(self):
        super(LayoutLMv3DocumentClassifierPreprocessorTest, self).setUp()
        self.preprocessor = LayoutLMv3DocumentClassifierPreprocessor(
            vocab_size=100,
            max_sequence_length=512,
            image_size=(112, 112),
        )
        
        # Create dummy inputs
        self.batch_size = 2
        self.text = ["This is a test document.", "Another test document."]
        self.bbox = [
            [[0, 0, 100, 100]] * len(text.split()) for text in self.text
        ]
        self.image = tf.random.uniform(
            (self.batch_size, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32
        )
    
    @test_util.run_in_graph_and_eager_modes
    def test_valid_call(self):
        """Test the preprocessor with valid inputs."""
        inputs = {
            "text": self.text,
            "bbox": self.bbox,
            "image": self.image,
        }
        outputs = self.preprocessor(inputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("bbox", outputs)
        self.assertIn("attention_mask", outputs)
        self.assertIn("image", outputs)
        self.assertEqual(outputs["input_ids"].shape, (self.batch_size, 512))
        self.assertEqual(outputs["bbox"].shape, (self.batch_size, 512, 4))
        self.assertEqual(outputs["attention_mask"].shape, (self.batch_size, 512))
        self.assertEqual(outputs["image"].shape, (self.batch_size, 112, 112, 3))
    
    @test_util.run_in_graph_and_eager_modes
    def test_save_and_load(self):
        """Test saving and loading the preprocessor."""
        inputs = {
            "text": self.text,
            "bbox": self.bbox,
            "image": self.image,
        }
        outputs = self.preprocessor(inputs)
        path = self.get_temp_dir()
        self.preprocessor.save(path)
        restored_preprocessor = tf.keras.models.load_model(path)
        restored_outputs = restored_preprocessor(inputs)
        self.assertAllClose(outputs["input_ids"], restored_outputs["input_ids"])
        self.assertAllClose(outputs["bbox"], restored_outputs["bbox"])
        self.assertAllClose(outputs["attention_mask"], restored_outputs["attention_mask"])
        self.assertAllClose(outputs["image"], restored_outputs["image"])
    
    @test_util.run_in_graph_and_eager_modes
    def test_from_preset(self):
        """Test creating a preprocessor from a preset."""
        preprocessor = LayoutLMv3DocumentClassifierPreprocessor.from_preset("layoutlmv3_base")
        inputs = {
            "text": ["Test document"],
            "bbox": [[[0, 0, 100, 100]] * 2],
            "image": tf.random.uniform((1, 112, 112, 3), dtype=tf.float32),
        }
        outputs = preprocessor(inputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("bbox", outputs)
        self.assertIn("attention_mask", outputs)
        self.assertIn("image", outputs)
    
    @test_util.run_in_graph_and_eager_modes
    def test_preprocessor_with_different_input_shapes(self):
        """Test the preprocessor with different input shapes."""
        # Test with different text lengths
        text_lengths = ["short", "a bit longer text", "a very very very long text that exceeds the maximum sequence length"]
        for text in text_lengths:
            inputs = {
                "text": [text],
                "bbox": [[[0, 0, 100, 100]] * len(text.split())],
                "image": tf.random.uniform((1, 112, 112, 3), dtype=tf.float32),
            }
            outputs = self.preprocessor(inputs)
            self.assertEqual(outputs["input_ids"].shape, (1, 512))
            self.assertEqual(outputs["bbox"].shape, (1, 512, 4))
            self.assertEqual(outputs["attention_mask"].shape, (1, 512))
        
        # Test with different batch sizes
        batch_sizes = [1, 4]
        for batch_size in batch_sizes:
            inputs = {
                "text": ["Test document"] * batch_size,
                "bbox": [[[0, 0, 100, 100]] * 2] * batch_size,
                "image": tf.random.uniform((batch_size, 112, 112, 3), dtype=tf.float32),
            }
            outputs = self.preprocessor(inputs)
            self.assertEqual(outputs["input_ids"].shape, (batch_size, 512))
            self.assertEqual(outputs["bbox"].shape, (batch_size, 512, 4))
            self.assertEqual(outputs["attention_mask"].shape, (batch_size, 512))
    
    @test_util.run_in_graph_and_eager_modes
    def test_preprocessor_with_invalid_inputs(self):
        """Test the preprocessor with invalid inputs."""
        # Test with empty text
        inputs = {
            "text": [""],
            "bbox": [[[0, 0, 100, 100]]],
            "image": tf.random.uniform((1, 112, 112, 3), dtype=tf.float32),
        }
        with self.assertRaises(ValueError):
            self.preprocessor(inputs)
        
        # Test with mismatched bbox and text lengths
        inputs = {
            "text": ["Test document"],
            "bbox": [[[0, 0, 100, 100]] * 3],  # More bboxes than words
            "image": tf.random.uniform((1, 112, 112, 3), dtype=tf.float32),
        }
        with self.assertRaises(ValueError):
            self.preprocessor(inputs)
        
        # Test with invalid image shape
        inputs = {
            "text": ["Test document"],
            "bbox": [[[0, 0, 100, 100]] * 2],
            "image": tf.random.uniform((1, 224, 224, 3), dtype=tf.float32),  # Wrong size
        }
        with self.assertRaises(ValueError):
            self.preprocessor(inputs) 