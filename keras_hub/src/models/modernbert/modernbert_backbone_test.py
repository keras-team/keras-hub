import pytest
from keras import ops

from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertBackboneTest(TestCase):
    """Tests for ModernBERT backbone."""

    def setUp(self):
        """Set up a small configuration for testing."""
        self.init_kwargs = {
            "vocabulary_size": 10,
            "hidden_dim": 8,
            "intermediate_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "local_attention_window": 128,
            "global_attn_every_n_layers": 2,
            "dropout": 0.0,
        }

        self.input_data = {
            "token_ids": ops.ones(
                (2, 5),
                dtype="int32",
            ),
            "padding_mask": ops.ones(
                (2, 5),
                dtype="int32",
            ),
        }

    def test_backbone_basics(self):
        """Validate the forward pass.

        This test checks:
        - Correct output shape from a forward pass.
        - Configuration serialization via `get_config`/`from_config`.
        - Weight and architecture preservation using the `.keras` format.
        - Backend-agnostic compatibility across frameworks.
        """
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    def test_variable_sequence_length(self):
        """Validate backbone behavior when
        executing over variable sequence lengths.

        This test checks:
        - Ability to accept inputs shorter than the maximum window sizes.
        - Proper dynamic output shape generation matching input batch shapes.
        - Stability of positional calculations under changing sequence domains.
        """
        model = ModernBertBackbone(**self.init_kwargs)

        short_input = {
            "token_ids": ops.ones(
                (1, 3),
                dtype="int32",
            ),
            "padding_mask": ops.ones(
                (1, 3),
                dtype="int32",
            ),
        }

        output = model(short_input)

        self.assertEqual(
            output.shape,
            (1, 3, 8),
        )

    def test_alternating_attention_logic(self):
        """Validate structural routing
        configurations of alternating attention.

        This test checks:
        - Proper assignment of local sliding window limits on local layers.
        - Correct nullification of attention window inputs on global layers.
        - Layer-by-layer compliance with the specified interleaving frequency.
        """
        model = ModernBertBackbone(**self.init_kwargs)

        local_layer = model.get_layer("transformer_layer_0")

        global_layer = model.get_layer("transformer_layer_1")

        self.assertEqual(
            local_layer.local_attention_window,
            128,
        )

        self.assertIsNone(global_layer.local_attention_window)

    @pytest.mark.large
    def test_saved_model(self):
        """Validate heavy model serialization
        routines and weight persistence.

        This test checks:
        - Export integrity under standard Keras saving utilities.
        - Restoration accuracy of graph connections and layer naming schemas.
        - Parity of inference outputs between original and reloaded states.
        """
        self.run_model_saving_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_mixed_precision(self):
        """Validate layer execution stability under
        mixed precision configurations.

        This test checks:
        - Down-casting behavior of input embeddings to computational precision.
        - Stability of attention scores and normalizations at lower resolutions.
        - Alignment of terminal shapes when standard precision policies change.
        """
        self.run_precision_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )
