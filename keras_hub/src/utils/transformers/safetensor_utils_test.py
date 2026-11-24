import json
import os
import tempfile

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

    def test_has_tensor_single_file_and_aliases(self):
        with SafetensorLoader(self.preset_dir) as loader:
            self.assertTrue(hasattr(loader, "has_tensor"))
            self.assertTrue(
                loader.has_tensor("encoder.layernorm_embedding.weight")
            )
            self.assertTrue(
                loader.has_tensor("model.decoder.embed_tokens.weight")
            )
            self.assertTrue(loader.has_tensor("shared.weight"))
            self.assertFalse(loader.has_tensor("encoder.missing.weight"))
            self.assertAllEqual(
                loader.get_tensor("encoder.layernorm_embedding.weight"),
                self.layer_norm,
            )

        custom_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        save_file(
            {"custom.weight": self.layer_norm},
            os.path.join(custom_dir, "custom.safetensors"),
        )
        with SafetensorLoader(custom_dir, fname="custom.safetensors") as loader:
            self.assertTrue(loader.has_tensor("custom.weight"))
            self.assertFalse(loader.has_tensor("absent.weight"))
            self.assertAllEqual(
                loader.get_tensor("custom.weight"), self.layer_norm
            )

        empty_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        with SafetensorLoader(empty_dir) as loader:
            self.assertFalse(loader.has_tensor("any.weight"))

    def test_has_tensor_preserves_prefix_and_later_lookups(self):
        tmp_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        aux_shared = np.array([1.0, 2.0], dtype="float32")
        aux_only = np.array([3.0, 4.0], dtype="float32")
        model_shared = np.array([5.0, 6.0], dtype="float32")
        save_file(
            {
                "aux.shared_suffix.weight": aux_shared,
                "aux.only_aux.weight": aux_only,
                "model.shared_suffix.weight": model_shared,
            },
            os.path.join(tmp_dir, "model.safetensors"),
        )
        with SafetensorLoader(tmp_dir, prefix="model.") as loader:
            self.assertTrue(hasattr(loader, "has_tensor"))
            self.assertFalse(loader.has_tensor("missing.weight"))
            self.assertEqual(loader.prefix, "model.")
            self.assertTrue(loader.has_tensor("only_aux.weight"))
            self.assertEqual(loader.prefix, "model.")
            self.assertAllEqual(
                loader.get_tensor("shared_suffix.weight"), model_shared
            )

        with SafetensorLoader(tmp_dir) as loader:
            self.assertTrue(loader.has_tensor("only_aux.weight"))
            self.assertIsNone(loader.prefix)
            self.assertFalse(loader.has_tensor("missing.weight"))
            self.assertIsNone(loader.prefix)

    def test_has_tensor_sharded_checkpoint(self):
        shard_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        shard_1 = "model-00001-of-00002.safetensors"
        shard_2 = "model-00002-of-00002.safetensors"
        save_file(
            {"model.layer0.weight": self.embedding},
            os.path.join(shard_dir, shard_1),
        )
        save_file(
            {"head.bias": self.layer_norm},
            os.path.join(shard_dir, shard_2),
        )
        with open(
            os.path.join(shard_dir, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(
                {
                    "weight_map": {
                        "model.layer0.weight": shard_1,
                        "head.bias": shard_2,
                    }
                },
                f,
            )

        with SafetensorLoader(shard_dir) as loader:
            self.assertTrue(hasattr(loader, "has_tensor"))
            self.assertFalse(loader.has_tensor("absent.weight"))
            self.assertEqual(loader.safetensor_files, {})
            self.assertIsNone(loader.prefix)
            self.assertTrue(loader.has_tensor("layer0.weight"))
            self.assertTrue(loader.has_tensor("model.layer0.weight"))
            self.assertTrue(loader.has_tensor("head.bias"))
            self.assertIsNone(loader.prefix)
            self.assertAllEqual(
                loader.get_tensor("layer0.weight"), self.embedding
            )
            self.assertAllEqual(loader.get_tensor("head.bias"), self.layer_norm)

    def test_has_tensor_missing_shard_or_explicit_fname_raises(self):
        shard_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        with open(
            os.path.join(shard_dir, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(
                {
                    "weight_map": {
                        "aux.pooler.dense.weight": (
                            "model-00001-of-00002.safetensors"
                        )
                    }
                },
                f,
            )
        with SafetensorLoader(shard_dir, prefix="model.") as loader:
            with self.assertRaises(FileNotFoundError):
                loader.has_tensor("pooler.dense.weight")
            self.assertEqual(loader.prefix, "model.")

        explicit_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        with SafetensorLoader(
            explicit_dir, prefix="model.", fname="missing.safetensors"
        ) as loader:
            with self.assertRaises(FileNotFoundError):
                loader.has_tensor("pooler.dense.weight")
            self.assertEqual(loader.prefix, "model.")
