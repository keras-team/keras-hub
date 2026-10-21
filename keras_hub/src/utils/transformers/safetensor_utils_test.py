import os

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import save_file

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader


class SafetensorLoaderTest(TestCase):
    def setUp(self):
        super().setUp()
        self.preset_dir = self.get_temp_dir()
        self.embedding = np.arange(6, dtype="float32").reshape(3, 2)
        self.layer_norm = np.array([0.5, 1.5], dtype="float32")
        # Mirror how `safetensors.torch.save_model` writes tied weights: the
        # tensor is stored once and the other names only appear in the
        # metadata as aliases of the stored key.
        save_file(
            {
                "model.decoder.embed_tokens.weight": self.embedding,
                "model.encoder.layernorm_embedding.weight": self.layer_norm,
            },
            os.path.join(self.preset_dir, "model.safetensors"),
            metadata={
                "format": "pt",
                "model.shared.weight": "model.decoder.embed_tokens.weight",
                "model.encoder.embed_tokens.weight": (
                    "model.decoder.embed_tokens.weight"
                ),
            },
        )

    def test_tied_weight_resolved_from_metadata(self):
        with SafetensorLoader(self.preset_dir) as loader:
            self.assertAllEqual(
                loader.get_tensor("shared.weight"), self.embedding
            )
            self.assertAllEqual(
                loader.get_tensor("encoder.embed_tokens.weight"),
                self.embedding,
            )
            self.assertAllEqual(
                loader.get_tensor("decoder.embed_tokens.weight"),
                self.embedding,
            )

    def test_stored_weight_still_resolved(self):
        with SafetensorLoader(self.preset_dir) as loader:
            self.assertAllEqual(
                loader.get_tensor("encoder.layernorm_embedding.weight"),
                self.layer_norm,
            )
            self.assertEqual(loader.prefix, "model.")

    def test_missing_weight_still_raises(self):
        with SafetensorLoader(self.preset_dir) as loader:
            with self.assertRaises(SafetensorError):
                loader.get_tensor("encoder.missing.weight")
