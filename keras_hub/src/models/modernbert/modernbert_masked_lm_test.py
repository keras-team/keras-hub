import pytest
import numpy as np
import keras
from keras import ops
from keras_hub.src.models.modernbert.modernbert_backbone import ModernBertBackbone
from keras_hub.src.models.modernbert.modernbert_masked_lm import ModernBertMaskedLM
from keras_hub.src.tests.test_case import TestCase

class ModernBertMaskedLMTest(TestCase):
    def setUp(self):
        # Small scale configuration for fast unit testing
        self.backbone = ModernBertBackbone(
            vocabulary_size=100,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
        )
        # Input: batch_size=2, sequence_length=5
        self.input_data = {
            "token_ids": ops.cast(np.ones((2, 5)), dtype="int32"),
            "padding_mask": ops.cast(np.ones((2, 5)), dtype="int32"),
        }
        
        # MLM Labels: -100 is the standard ignore index for CrossEntropy
        self.y_true = np.full((2, 5), -100, dtype="int32")
        self.y_true[:, 1:3] = np.random.randint(0, 100, (2, 2))
        
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": None, # Standard for Keras Hub model tests
        }

    def test_task_basics(self):
        """
        Test forward pass, output shapes, and basic serialization.
        """
        self.run_task_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 100),
        )

    def test_compute_loss(self):
        """
        Verify that loss ignores -100 labels and doesn't produce NaN.
        """
        task = ModernBertMaskedLM(**self.init_kwargs)
        
        # Mock logits
        y_pred = keras.random.uniform((2, 5, 100))
        
        loss = task.compute_loss(x=self.input_data, y=self.y_true, y_pred=y_pred)
        
        self.assertNotAllClose(loss, 0.0)
        self.assertFalse(ops.any(ops.isnan(loss)))

    @pytest.mark.large
    def test_saved_model(self):
        """
        Verify the Task can be saved and loaded (serialization).
        """
        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_weight_tying(self):
        """
        Verify that the MLM head is using the backbone's embedding weights.
        """
        # Most Keras Hub models use tie_weights=True by default in their Task classes
        task = ModernBertMaskedLM(**self.init_kwargs)
        
        # Check if the output projection layer is actually using the embedding layer's variable
        embedding_weights = self.backbone.get_layer("token_embedding").embeddings
        
        # Verify the Task does not create its own separate kernel for vocab projection
        # This checks the 'prediction_head' layer within the MLM Task
        trainable_variables = [v.name for v in task.trainable_variables]
        self.assertFalse(any("prediction_head/kernel" in name for name in trainable_variables))
        
        # Verify the backbone embedding is indeed in the task variables
        self.assertTrue(any("token_embedding/embeddings" in name for name in trainable_variables))

    def test_fit(self):
        """
        Quick sanity check that the model can perform a train step.
        """
        task = ModernBertMaskedLM(**self.init_kwargs)
        task.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        task.fit(x=self.input_data, y=self.y_true, batch_size=2, epochs=1)