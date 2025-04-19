import pytest
from keras import ops

from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.tests.test_case import TestCase


class DeiTBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "image_shape": (28, 28, 3),
            "patch_size": 4,
            "num_layers": 3,
            "hidden_dim": 48,
            "num_heads": 6,
            "intermediate_dim": 48 * 4,
            "use_mha_bias": True,
        }
        self.input_size = 28
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=DeiTBackbone,
            init_kwargs={**self.init_kwargs},
            input_data=self.input_data,
            # 49+2 positions(49 patches, cls and distillation token)
            expected_output_shape=(2, 51, 48),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DeiTBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
