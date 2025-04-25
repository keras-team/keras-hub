"""Tests for LayoutLMv3 backbone."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from ..layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone

class LayoutLMv3BackboneTest(tf.test.TestCase):
    def setUp(self):
        super(LayoutLMv3BackboneTest, self).setUp()
        self.backbone = LayoutLMv3Backbone(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            image_size=(112, 112),
            patch_size=16,
        )
        
        # Create dummy inputs
        self.batch_size = 2
        self.seq_length = 16
        self.input_ids = tf.random.uniform(
            (self.batch_size, self.seq_length), minval=0, maxval=100, dtype=tf.int32
        )
        self.bbox = tf.random.uniform(
            (self.batch_size, self.seq_length, 4), minval=0, maxval=100, dtype=tf.int32
        )
        self.attention_mask = tf.ones((self.batch_size, self.seq_length), dtype=tf.int32)
        self.image = tf.random.uniform(
            (self.batch_size, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32
        )
        
        self.inputs = {
            "input_ids": self.input_ids,
            "bbox": self.bbox,
            "attention_mask": self.attention_mask,
            "image": self.image,
        }
    
    @test_util.run_in_graph_and_eager_modes
    def test_valid_call(self):
        """Test the backbone with valid inputs."""
        outputs = self.backbone(self.inputs)
        self.assertIn("sequence_output", outputs)
        self.assertIn("pooled_output", outputs)
        self.assertEqual(outputs["sequence_output"].shape, (self.batch_size, self.seq_length + 49 + 1, 64))  # text + image patches + cls
        self.assertEqual(outputs["pooled_output"].shape, (self.batch_size, 64))
    
    @test_util.run_in_graph_and_eager_modes
    def test_save_and_load(self):
        """Test saving and loading the backbone."""
        outputs = self.backbone(self.inputs)
        path = self.get_temp_dir()
        self.backbone.save(path)
        restored_backbone = tf.keras.models.load_model(path)
        restored_outputs = restored_backbone(self.inputs)
        self.assertAllClose(outputs["sequence_output"], restored_outputs["sequence_output"])
        self.assertAllClose(outputs["pooled_output"], restored_outputs["pooled_output"])
    
    @test_util.run_in_graph_and_eager_modes
    def test_from_preset(self):
        """Test creating a backbone from a preset."""
        backbone = LayoutLMv3Backbone.from_preset("layoutlmv3_base")
        inputs = {
            "input_ids": tf.random.uniform((2, 16), 0, 100, dtype=tf.int32),
            "bbox": tf.random.uniform((2, 16, 4), 0, 100, dtype=tf.int32),
            "attention_mask": tf.ones((2, 16), dtype=tf.int32),
            "image": tf.random.uniform((2, 112, 112, 3), dtype=tf.float32),
        }
        outputs = backbone(inputs)
        self.assertIn("sequence_output", outputs)
        self.assertIn("pooled_output", outputs)
        
    @test_util.run_in_graph_and_eager_modes
    def test_backbone_with_different_input_shapes(self):
        """Test the backbone with different input shapes."""
        # Test with different sequence lengths
        seq_lengths = [32, 128]
        for seq_len in seq_lengths:
            inputs = {
                "input_ids": tf.random.uniform(
                    (self.batch_size, seq_len), minval=0, maxval=100, dtype=tf.int32
                ),
                "bbox": tf.random.uniform(
                    (self.batch_size, seq_len, 4), minval=0, maxval=100, dtype=tf.int32
                ),
                "attention_mask": tf.ones((self.batch_size, seq_len), dtype=tf.int32),
                "image": self.image,
            }
            outputs = self.backbone(inputs)
            expected_seq_length = seq_len + 49 + 1
            self.assertEqual(outputs["sequence_output"].shape, (self.batch_size, expected_seq_length, 64))
        
        # Test with different batch sizes
        batch_sizes = [1, 4]
        for batch_size in batch_sizes:
            inputs = {
                "input_ids": tf.random.uniform(
                    (batch_size, self.seq_length), minval=0, maxval=100, dtype=tf.int32
                ),
                "bbox": tf.random.uniform(
                    (batch_size, self.seq_length, 4), minval=0, maxval=100, dtype=tf.int32
                ),
                "attention_mask": tf.ones((batch_size, self.seq_length), dtype=tf.int32),
                "image": tf.random.uniform(
                    (batch_size, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32
                ),
            }
            outputs = self.backbone(inputs)
            expected_seq_length = self.seq_length + 49 + 1
            self.assertEqual(outputs["sequence_output"].shape, (batch_size, expected_seq_length, 64))
    
    @test_util.run_in_graph_and_eager_modes
    def test_backbone_with_attention_mask(self):
        """Test the backbone with different attention masks."""
        # Create a mask with some padding
        attention_mask = tf.ones((self.batch_size, self.seq_length), dtype=tf.int32)
        attention_mask = tf.tensor_scatter_nd_update(
            attention_mask,
            tf.constant([[0, 32], [1, 48]]),  # Set some positions to 0
            tf.constant([0, 0], dtype=tf.int32),
        )
        
        inputs = {
            "input_ids": self.input_ids,
            "bbox": self.bbox,
            "attention_mask": attention_mask,
            "image": self.image,
        }
        
        outputs = self.backbone(inputs)
        self.assertIsInstance(outputs, dict)
        self.assertIn("sequence_output", outputs)
        self.assertIn("pooled_output", outputs)
    
    @test_util.run_in_graph_and_eager_modes
    def test_backbone_gradient(self):
        """Test that the backbone produces gradients."""
        with tf.GradientTape() as tape:
            outputs = self.backbone(self.inputs)
            loss = tf.reduce_mean(outputs["pooled_output"])
        
        # Check if gradients exist for all trainable variables
        gradients = tape.gradient(loss, self.backbone.trainable_variables)
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(tf.reduce_all(tf.math.is_nan(grad)))
            self.assertFalse(tf.reduce_all(tf.math.is_inf(grad))) 