import pytest
from keras import ops

from keras_hub.src.models.electra.electra_backbone import ElectraBackbone
from keras_hub.src.tests.test_case import TestCase


class ElectraBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocab_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "embedding_dim": 2,
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
            cls=ElectraBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 5, 2),
                "pooled_output": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ElectraBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=ElectraBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=ElectraBackbone,
            preset="electra_small_discriminator_uncased_en",
            input_data={
                "token_ids": ops.array([[101, 1996, 4248, 102]], dtype="int32"),
                "segment_ids": ops.zeros((1, 4), dtype="int32"),
                "padding_mask": ops.ones((1, 4), dtype="int32"),
            },
            expected_output_shape={
                "sequence_output": (1, 4, 256),
                "pooled_output": (1, 256),
            },
            # The forward pass from a preset should be stable!
            expected_partial_output={
                "sequence_output": (
                    ops.array([0.32287, 0.18754, -0.22272, -0.24177, 1.18977])
                ),
                "pooled_output": (
                    ops.array([-0.02974, 0.23383, 0.08430, -0.19471, 0.14822])
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ElectraBackbone.presets:
            self.run_preset_test(
                cls=ElectraBackbone,
                preset=preset,
                input_data=self.input_data,
            )
