import json
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

    def test_alias_map_skips_format_and_stored_keys(self):
        save_file(
            {
                "model.decoder.embed_tokens.weight": self.embedding,
                "model.encoder.layernorm_embedding.weight": self.layer_norm,
            },
            os.path.join(self.preset_dir, "model.safetensors"),
            metadata={
                "format": "pt",
                "model.shared.weight": "model.decoder.embed_tokens.weight",
                # A metadata key that is itself a stored tensor is not an
                # alias and must not shadow the stored weight.
                "model.encoder.layernorm_embedding.weight": (
                    "model.decoder.embed_tokens.weight"
                ),
            },
        )
        with SafetensorLoader(self.preset_dir) as loader:
            self.assertAllEqual(
                loader.get_tensor("shared.weight"), self.embedding
            )
            self.assertEqual(
                loader.safetensor_aliases["model.safetensors"],
                {"model.shared.weight": "model.decoder.embed_tokens.weight"},
            )
            self.assertAllEqual(
                loader.get_tensor("encoder.layernorm_embedding.weight"),
                self.layer_norm,
            )

    def write_index(self, name, weight_map):
        with open(os.path.join(self.preset_dir, name), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)

    def test_config_file_selects_the_index(self):
        save_file(
            {"model.norm.weight": self.layer_norm},
            os.path.join(self.preset_dir, "shard.safetensors"),
        )
        self.write_index(
            "other.safetensors.index.json",
            {"model.norm.weight": "shard.safetensors"},
        )
        with SafetensorLoader(
            self.preset_dir, config_file="other.safetensors.index.json"
        ) as loader:
            self.assertAllEqual(
                loader.get_tensor("norm.weight"), self.layer_norm
            )

    def test_config_file_defaults_to_the_root_index(self):
        self.write_index(
            "model.safetensors.index.json",
            {"model.decoder.embed_tokens.weight": "model.safetensors"},
        )
        with SafetensorLoader(self.preset_dir) as loader:
            self.assertIsNotNone(loader.safetensor_config)
        with self.assertRaisesRegex(ValueError, "Cannot specify `fname`"):
            SafetensorLoader(self.preset_dir, fname="model.safetensors")

    def test_config_file_none_ignores_the_root_index(self):
        self.write_index(
            "model.safetensors.index.json",
            {"model.other.weight": "missing.safetensors"},
        )
        with SafetensorLoader(
            self.preset_dir, fname="model.safetensors", config_file=None
        ) as loader:
            self.assertIsNone(loader.safetensor_config)
            self.assertAllEqual(
                loader.get_tensor("encoder.layernorm_embedding.weight"),
                self.layer_norm,
            )
