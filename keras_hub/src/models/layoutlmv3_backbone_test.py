# Copyright 2024 The Keras Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import numpy as np
from keras import testing_utils
from keras import ops
from keras import backend
from keras.testing import test_case
from ..layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone

class LayoutLMv3BackboneTest(test_case.TestCase):
    def setUp(self):
        super().setUp()
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
        self.input_ids = ops.random.uniform(
            (self.batch_size, self.seq_length), minval=0, maxval=100, dtype="int32"
        )
        self.bbox = ops.random.uniform(
            (self.batch_size, self.seq_length, 4), minval=0, maxval=100, dtype="int32"
        )
        self.attention_mask = ops.ones((self.batch_size, self.seq_length), dtype="int32")
        self.image = ops.random.uniform(
            (self.batch_size, 112, 112, 3), minval=0, maxval=1, dtype="float32"
        )
        
        self.inputs = {
            "input_ids": self.input_ids,
            "bbox": self.bbox,
            "attention_mask": self.attention_mask,
            "image": self.image,
        }
    
    def test_valid_call(self):
        """Test the backbone with valid inputs."""
        outputs = self.backbone(self.inputs)
        self.assertIn("sequence_output", outputs)
        self.assertIn("pooled_output", outputs)
        self.assertEqual(outputs["sequence_output"].shape, (self.batch_size, self.seq_length + 49 + 1, 64))  # text + image patches + cls
        self.assertEqual(outputs["pooled_output"].shape, (self.batch_size, 64))
    
    def test_save_and_load(self):
        """Test saving and loading the backbone."""
        outputs = self.backbone(self.inputs)
        path = self.get_temp_dir()
        self.backbone.save(path)
        restored_backbone = backend.saving.load_model(path)
        restored_outputs = restored_backbone(self.inputs)
        self.assertAllClose(outputs["sequence_output"], restored_outputs["sequence_output"])
        self.assertAllClose(outputs["pooled_output"], restored_outputs["pooled_output"])
    
    def test_from_preset(self):
        """Test creating a backbone from a preset."""
        backbone = LayoutLMv3Backbone.from_preset("layoutlmv3_base")
        inputs = {
            "input_ids": ops.random.uniform((2, 16), 0, 100, dtype="int32"),
            "bbox": ops.random.uniform((2, 16, 4), 0, 100, dtype="int32"),
            "attention_mask": ops.ones((2, 16), dtype="int32"),
            "image": ops.random.uniform((2, 112, 112, 3), dtype="float32"),
        }
        outputs = backbone(inputs)
        self.assertIn("sequence_output", outputs)
        self.assertIn("pooled_output", outputs)
        
    def test_backbone_with_different_input_shapes(self):
        """Test the backbone with different input shapes."""
        # Test with different sequence lengths
        seq_lengths = [32, 128]
        for seq_len in seq_lengths:
            inputs = {
                "input_ids": ops.random.uniform(
                    (self.batch_size, seq_len), minval=0, maxval=100, dtype="int32"
                ),
                "bbox": ops.random.uniform(
                    (self.batch_size, seq_len, 4), minval=0, maxval=100, dtype="int32"
                ),
                "attention_mask": ops.ones((self.batch_size, seq_len), dtype="int32"),
                "image": self.image,
            }
            outputs = self.backbone(inputs)
            expected_seq_length = seq_len + 49 + 1
            self.assertEqual(outputs["sequence_output"].shape, (self.batch_size, expected_seq_length, 64))
        
        # Test with different batch sizes
        batch_sizes = [1, 4]
        for batch_size in batch_sizes:
            inputs = {
                "input_ids": ops.random.uniform(
                    (batch_size, self.seq_length), minval=0, maxval=100, dtype="int32"
                ),
                "bbox": ops.random.uniform(
                    (batch_size, self.seq_length, 4), minval=0, maxval=100, dtype="int32"
                ),
                "attention_mask": ops.ones((batch_size, self.seq_length), dtype="int32"),
                "image": ops.random.uniform(
                    (batch_size, 112, 112, 3), minval=0, maxval=1, dtype="float32"
                ),
            }
            outputs = self.backbone(inputs)
            expected_seq_length = self.seq_length + 49 + 1
            self.assertEqual(outputs["sequence_output"].shape, (batch_size, expected_seq_length, 64))
    
    def test_backbone_with_attention_mask(self):
        """Test the backbone with different attention masks."""
        # Create a mask with some padding
        attention_mask = ops.ones((self.batch_size, self.seq_length), dtype="int32")
        indices = ops.array([[0, 32], [1, 48]], dtype="int32")
        updates = ops.array([0, 0], dtype="int32")
        attention_mask = ops.scatter_nd(indices, updates, attention_mask.shape)
        
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
        with backend.GradientTape() as tape:
            outputs = self.backbone(self.inputs)
            loss = ops.mean(outputs["pooled_output"])
        
        # Check if gradients exist for all trainable variables
        gradients = tape.gradient(loss, self.backbone.trainable_variables)
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(ops.all(ops.isnan(grad)))
            self.assertFalse(ops.all(ops.isinf(grad))) 