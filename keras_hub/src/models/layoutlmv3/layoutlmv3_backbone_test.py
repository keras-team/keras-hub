import os
import pytest
import tensorflow as tf
import numpy as np
from keras import backend
from tensorflow.python.keras.testing_utils import test_combinations
from tensorflow.python.keras.testing_utils import test_utils
from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone

@test_combinations.run_all_keras_modes
class LayoutLMv3BackboneTest(test_combinations.TestCase):
    def setUp(self):
        super(LayoutLMv3BackboneTest, self).setUp()
        self.backbone = LayoutLMv3Backbone(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=(112, 112),
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=False,
            rel_pos_bins=32,
            max_rel_pos=128,
        )
        
        # Create dummy inputs
        self.batch_size = 2
        self.seq_length = 64
        self.input_ids = tf.random.uniform(
            (self.batch_size, self.seq_length), minval=0, maxval=30522, dtype=tf.int32
        )
        self.bbox = tf.random.uniform(
            (self.batch_size, self.seq_length, 4), minval=0, maxval=512, dtype=tf.int32
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
    
    def test_backbone_basics(self):
        """Test the basic functionality of the backbone."""
        # Test model creation
        self.assertIsInstance(self.backbone, LayoutLMv3Backbone)
        
        # Test model call
        outputs = self.backbone(self.inputs)
        self.assertIsInstance(outputs, dict)
        self.assertIn("sequence_output", outputs)
        self.assertIn("pooled_output", outputs)
        
        # Test output shapes
        sequence_output = outputs["sequence_output"]
        pooled_output = outputs["pooled_output"]
        
        expected_seq_length = self.seq_length + (112 // 16) * (112 // 16) + 1  # text + image patches + cls token
        self.assertEqual(sequence_output.shape, (self.batch_size, expected_seq_length, 768))
        self.assertEqual(pooled_output.shape, (self.batch_size, 768))
    
    def test_backbone_save_and_load(self):
        """Test saving and loading the backbone."""
        # Save the model
        save_path = os.path.join(self.get_temp_dir(), "layoutlmv3_backbone")
        self.backbone.save(save_path)
        
        # Load the model
        loaded_backbone = tf.keras.models.load_model(save_path)
        
        # Test loaded model
        outputs = loaded_backbone(self.inputs)
        self.assertIsInstance(outputs, dict)
        self.assertIn("sequence_output", outputs)
        self.assertIn("pooled_output", outputs)
        
        # Compare outputs
        original_outputs = self.backbone(self.inputs)
        tf.debugging.assert_near(
            outputs["sequence_output"], original_outputs["sequence_output"], rtol=1e-5
        )
        tf.debugging.assert_near(
            outputs["pooled_output"], original_outputs["pooled_output"], rtol=1e-5
        )
    
    def test_backbone_with_different_input_shapes(self):
        """Test the backbone with different input shapes."""
        # Test with different sequence lengths
        seq_lengths = [32, 128]
        for seq_len in seq_lengths:
            inputs = {
                "input_ids": tf.random.uniform(
                    (self.batch_size, seq_len), minval=0, maxval=30522, dtype=tf.int32
                ),
                "bbox": tf.random.uniform(
                    (self.batch_size, seq_len, 4), minval=0, maxval=512, dtype=tf.int32
                ),
                "attention_mask": tf.ones((self.batch_size, seq_len), dtype=tf.int32),
                "image": self.image,
            }
            outputs = self.backbone(inputs)
            expected_seq_length = seq_len + (112 // 16) * (112 // 16) + 1
            self.assertEqual(outputs["sequence_output"].shape, (self.batch_size, expected_seq_length, 768))
        
        # Test with different batch sizes
        batch_sizes = [1, 4]
        for batch_size in batch_sizes:
            inputs = {
                "input_ids": tf.random.uniform(
                    (batch_size, self.seq_length), minval=0, maxval=30522, dtype=tf.int32
                ),
                "bbox": tf.random.uniform(
                    (batch_size, self.seq_length, 4), minval=0, maxval=512, dtype=tf.int32
                ),
                "attention_mask": tf.ones((batch_size, self.seq_length), dtype=tf.int32),
                "image": tf.random.uniform(
                    (batch_size, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32
                ),
            }
            outputs = self.backbone(inputs)
            expected_seq_length = self.seq_length + (112 // 16) * (112 // 16) + 1
            self.assertEqual(outputs["sequence_output"].shape, (batch_size, expected_seq_length, 768))
    
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