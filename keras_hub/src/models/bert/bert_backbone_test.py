import pytest
from keras import ops

from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.tests.test_case import TestCase


class BertBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "segment_ids": ops.zeros((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=BertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 5, 2),
                "pooled_output": (2, 2),
            },
        )

    def test_backbone_with_dora(self):
        """Test BERT backbone with DoRA layers enabled."""
        dora_init_kwargs = {
            **self.init_kwargs,
            "enable_dora": True,
            "dora_rank": 4,
            "dora_alpha": 8.0,
        }

        self.run_backbone_test(
            cls=BertBackbone,
            init_kwargs=dora_init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 5, 2),
                "pooled_output": (2, 2),
            },
        )

    def test_dora_config_preservation(self):
        """Test that DoRA configuration is properly saved and restored."""
        model = BertBackbone(
            vocabulary_size=10,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            enable_dora=True,
            dora_rank=8,
            dora_alpha=16.0,
            max_sequence_length=5,
        )

        config = model.get_config()

        # Verify DoRA parameters are in config
        self.assertEqual(config["enable_dora"], True)
        self.assertEqual(config["dora_rank"], 8)
        self.assertEqual(config["dora_alpha"], 16.0)

        # Test model can be recreated from config
        new_model = BertBackbone.from_config(config)
        self.assertEqual(new_model.enable_dora, True)
        self.assertEqual(new_model.dora_rank, 8)
        self.assertEqual(new_model.dora_alpha, 16.0)

    def test_dora_vs_regular_output_shapes(self):
        """Test that DoRA and regular models produce same output shapes."""
        regular_model = BertBackbone(**self.init_kwargs)
        dora_model = BertBackbone(
            **self.init_kwargs,
            enable_dora=True,
            dora_rank=4,
            dora_alpha=8.0,
        )

        regular_output = regular_model(self.input_data)
        dora_output = dora_model(self.input_data)

        # Shapes should be identical
        self.assertEqual(
            regular_output["sequence_output"].shape,
            dora_output["sequence_output"].shape,
        )
        self.assertEqual(
            regular_output["pooled_output"].shape,
            dora_output["pooled_output"].shape,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_saved_model_with_dora(self):
        """Test model saving/loading with DoRA enabled."""
        dora_init_kwargs = {
            **self.init_kwargs,
            "enable_dora": True,
            "dora_rank": 4,
            "dora_alpha": 8.0,
        }

        self.run_model_saving_test(
            cls=BertBackbone,
            init_kwargs=dora_init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BertBackbone,
            preset="bert_tiny_en_uncased",
            input_data={
                "token_ids": ops.array([[101, 1996, 4248, 102]], dtype="int32"),
                "segment_ids": ops.zeros((1, 4), dtype="int32"),
                "padding_mask": ops.ones((1, 4), dtype="int32"),
            },
            expected_output_shape={
                "sequence_output": (1, 4, 128),
                "pooled_output": (1, 128),
            },
            # The forward pass from a preset should be stable!
            expected_partial_output={
                "sequence_output": (
                    ops.array([-1.38173, 0.16598, -2.92788, -2.66958, -0.61556])
                ),
                "pooled_output": (
                    ops.array([-0.99999, 0.07777, -0.99955, -0.00982, -0.99967])
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BertBackbone.presets:
            self.run_preset_test(
                cls=BertBackbone,
                preset=preset,
                input_data=self.input_data,
            )
