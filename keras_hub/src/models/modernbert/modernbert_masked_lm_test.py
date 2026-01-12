import pytest
import numpy as np
import keras
from keras import ops
from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modernbert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.tests.test_case import TestCase

class ModernBertMaskedLMTest(TestCase):
    def setUp(self):
        self.backbone = ModernBertBackbone(
            vocabulary_size=100,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
        )
        self.input_data = {
            "token_ids": ops.cast(np.ones((2, 5)), dtype="int32"),
            "padding_mask": ops.cast(np.ones((2, 5)), dtype="int32"),
            "mask_positions": ops.cast(np.zeros((2, 5)), dtype="int32"),
        }
        self.y_true = np.random.randint(0, 100, (2, 5))

        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": None,
        }

    def test_task_basics(self):
        """Test forward pass and output shape."""
        self.run_task_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 100),
        )

    def test_weight_tying(self):
        """
        Verify that the MLM head shares the embedding matrix kernel.
        """
        task = ModernBertMaskedLM(**self.init_kwargs)

        _ = task(self.input_data)
        
        # Find the embedding variable and the head's kernel
        embedding_matrix = self.backbone.get_layer("token_embedding").embeddings
        
        # In MaskedLMHead, the kernel is set to the embedding weights. 
        # object IDs or memory addresses are identical.
        self.assertIs(task.mlm_head.kernel, embedding_matrix)
        
        trainable_vars = [v.name for v in task.trainable_variables]
        self.assertFalse(any("prediction_head/kernel" in name for name in trainable_vars))

    def test_fit(self):
        """Test a single training step."""
        task = ModernBertMaskedLM(**self.init_kwargs)
        task.compile(
            optimizer="adam", 
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        # Train on a single batch
        task.fit(x=self.input_data, y=self.y_true, batch_size=2, epochs=1)

    @pytest.mark.large
    def test_serialization(self):
        """
        Verify the model can be saved and reloaded.
        """
        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )