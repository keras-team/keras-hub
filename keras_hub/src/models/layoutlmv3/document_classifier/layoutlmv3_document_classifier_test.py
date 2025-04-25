"""Tests for LayoutLMv3 document classifier."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from ..layoutlmv3.layoutlmv3_document_classifier import LayoutLMv3DocumentClassifier

class LayoutLMv3DocumentClassifierTest(tf.test.TestCase):
    def setUp(self):
        super(LayoutLMv3DocumentClassifierTest, self).setUp()
        self.classifier = LayoutLMv3DocumentClassifier(
            num_classes=2,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            max_2d_position_embeddings=1024,
            image_size=112,
            patch_size=16,
            num_channels=3,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
        )
        
        # Create dummy inputs
        self.batch_size = 2
        self.input_ids = tf.random.uniform(
            (self.batch_size, 512), minval=0, maxval=100, dtype=tf.int32
        )
        self.bbox = tf.random.uniform(
            (self.batch_size, 512, 4), minval=0, maxval=1000, dtype=tf.int32
        )
        self.attention_mask = tf.ones((self.batch_size, 512), dtype=tf.int32)
        self.image = tf.random.uniform(
            (self.batch_size, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32
        )
    
    @test_util.run_in_graph_and_eager_modes
    def test_valid_call(self):
        """Test the classifier with valid inputs."""
        inputs = {
            "input_ids": self.input_ids,
            "bbox": self.bbox,
            "attention_mask": self.attention_mask,
            "image": self.image,
        }
        outputs = self.classifier(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, 2))
    
    @test_util.run_in_graph_and_eager_modes
    def test_save_and_load(self):
        """Test saving and loading the classifier."""
        inputs = {
            "input_ids": self.input_ids,
            "bbox": self.bbox,
            "attention_mask": self.attention_mask,
            "image": self.image,
        }
        outputs = self.classifier(inputs)
        path = self.get_temp_dir()
        self.classifier.save(path)
        restored_classifier = tf.keras.models.load_model(path)
        restored_outputs = restored_classifier(inputs)
        self.assertAllClose(outputs, restored_outputs)
    
    @test_util.run_in_graph_and_eager_modes
    def test_from_preset(self):
        """Test creating a classifier from a preset."""
        classifier = LayoutLMv3DocumentClassifier.from_preset("layoutlmv3_base", num_classes=2)
        inputs = {
            "input_ids": tf.random.uniform((1, 512), minval=0, maxval=100, dtype=tf.int32),
            "bbox": tf.random.uniform((1, 512, 4), minval=0, maxval=1000, dtype=tf.int32),
            "attention_mask": tf.ones((1, 512), dtype=tf.int32),
            "image": tf.random.uniform((1, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32),
        }
        outputs = classifier(inputs)
        self.assertEqual(outputs.shape, (1, 2))
    
    @test_util.run_in_graph_and_eager_modes
    def test_classifier_with_different_input_shapes(self):
        """Test the classifier with different input shapes."""
        # Test with different batch sizes
        batch_sizes = [1, 4]
        for batch_size in batch_sizes:
            inputs = {
                "input_ids": tf.random.uniform((batch_size, 512), minval=0, maxval=100, dtype=tf.int32),
                "bbox": tf.random.uniform((batch_size, 512, 4), minval=0, maxval=1000, dtype=tf.int32),
                "attention_mask": tf.ones((batch_size, 512), dtype=tf.int32),
                "image": tf.random.uniform((batch_size, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32),
            }
            outputs = self.classifier(inputs)
            self.assertEqual(outputs.shape, (batch_size, 2))
    
    @test_util.run_in_graph_and_eager_modes
    def test_classifier_with_invalid_inputs(self):
        """Test the classifier with invalid inputs."""
        # Test with wrong input shapes
        inputs = {
            "input_ids": tf.random.uniform((2, 256), minval=0, maxval=100, dtype=tf.int32),  # Wrong sequence length
            "bbox": tf.random.uniform((2, 512, 4), minval=0, maxval=1000, dtype=tf.int32),
            "attention_mask": tf.ones((2, 512), dtype=tf.int32),
            "image": tf.random.uniform((2, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32),
        }
        with self.assertRaises(ValueError):
            self.classifier(inputs)
        
        # Test with wrong image shape
        inputs = {
            "input_ids": tf.random.uniform((2, 512), minval=0, maxval=100, dtype=tf.int32),
            "bbox": tf.random.uniform((2, 512, 4), minval=0, maxval=1000, dtype=tf.int32),
            "attention_mask": tf.ones((2, 512), dtype=tf.int32),
            "image": tf.random.uniform((2, 224, 224, 3), minval=0, maxval=1, dtype=tf.float32),  # Wrong size
        }
        with self.assertRaises(ValueError):
            self.classifier(inputs) 