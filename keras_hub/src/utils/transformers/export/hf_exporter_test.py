import os
from unittest import mock

import keras
from keras import ops
from safetensors.numpy import load_file

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.gpt2 import get_gpt2_weights_map
from keras_hub.src.utils.transformers.export.hf_exporter import export_backbone


class ExportBackboneTest(TestCase):
    def test_export_backbone_unlisted_backend(self):
        backbone = GPT2Backbone(
            vocabulary_size=10,
            num_layers=1,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=8,
        )
        path = os.path.join(self.get_temp_dir(), "export")
        # A backend name that the exporter does not know.
        with mock.patch.object(
            keras.config, "backend", return_value="fakebackend"
        ):
            export_backbone(backbone, path)

        weights = load_file(os.path.join(path, "model.safetensors"))
        self.assertEqual(
            sorted(weights), sorted(get_gpt2_weights_map(backbone))
        )
        self.assertAllEqual(
            weights["transformer.wte.weight"],
            ops.convert_to_numpy(
                backbone.get_layer("token_embedding").embeddings
            ),
        )
