# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.gemma.gemma_backbone import GemmaBackbone
from keras_nlp.tests.test_case import TestCase


@pytest.mark.keras_3_only
class GemmaBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 256128,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 256,
            "head_dim": 128,
            "layer_norm_epsilon": 1e-6,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 128),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=GemmaBackbone,
            preset="gemma_2b_en",
            input_data={
                "token_ids": ops.array([[651, 4320, 8426, 25341, 235265]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 2048),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [1.073359, 0.262374, 0.170238, 0.605402, 2.336161]
            ),
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GemmaBackbone.presets:
            self.run_preset_test(
                cls=GemmaBackbone,
                preset=preset,
                input_data=self.input_data,
            )

    def test_architecture_characteristics(self):
        model = GemmaBackbone(**self.init_kwargs)
        self.assertEqual(model.count_params(), 33407616)
        self.assertEqual(len(model.layers), 6)

    def test_distribution(self):
        if keras.backend.backend() != "jax":
            return
        devices = keras.distribution.list_devices("CPU")
        if len(devices) == 1:
            # Need more than 1 device for distribution testing.
            return
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, len(devices)),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = GemmaBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(device_mesh, layout_map)
        with distribution.scope():
            model = GemmaBackbone(**self.init_kwargs)

        for w in model.weights:
            if "token_embedding/embeddings" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), (None, "model"))
            if "attention/query/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, "model", None)
                )
            if "attention/key/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, "model", None)
                )
            if "attention/value/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, "model", None)
                )
            if "attention/attention_output/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, None, "model")
                )
            if "ffw_gating/kernel" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), ("model", None))
            if "ffw_gating_2/kernel" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), ("model", None))
            if "ffw_linearl" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), (None, "model"))
