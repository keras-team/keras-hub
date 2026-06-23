import os

import h5py
import keras
import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import METADATA_FILE
from keras_hub.src.utils.preset_utils import MODEL_WEIGHTS_FILE
from keras_hub.src.utils.preset_utils import check_config_class
from keras_hub.src.utils.preset_utils import load_json


class TestBackbone(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertBackbone.presets.keys())
        gpt2_presets = set(GPT2Backbone.presets.keys())
        all_presets = set(Backbone.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)
        self.assertIn("bert_tiny_en_uncased", bert_presets)
        self.assertNotIn("bert_tiny_en_uncased", gpt2_presets)
        self.assertIn("gpt2_base_en", gpt2_presets)
        self.assertNotIn("gpt2_base_en", bert_presets)
        self.assertIn("bert_tiny_en_uncased", all_presets)
        self.assertIn("gpt2_base_en", all_presets)

    def test_lora_save_and_reload(self):
        # Regression test for #2701: saving LoRA weights and reloading them
        # into a freshly constructed model must reproduce the same outputs.
        init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "segment_ids": np.zeros((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        backbone = BertBackbone(**init_kwargs)
        # Snapshot the base weights so the reloaded model starts from an
        # identical state; only the LoRA adapters travel through `.lora.h5`.
        base_path = os.path.join(self.get_temp_dir(), "base.weights.h5")
        backbone.save_weights(base_path)

        backbone.enable_lora(4)
        # Simulate fine-tuning: give the (initially zero) `lora_kernel_b`
        # adapters non-zero values so the LoRA branch changes the output.
        rng = np.random.RandomState(42)
        for layer in backbone._flatten_layers():
            if getattr(layer, "lora_kernel_a", None) is not None:
                layer.lora_kernel_a.assign(
                    rng.normal(size=layer.lora_kernel_a.shape).astype("float32")
                )
                layer.lora_kernel_b.assign(
                    rng.normal(size=layer.lora_kernel_b.shape).astype("float32")
                )
        lora_path = os.path.join(self.get_temp_dir(), "model.lora.h5")
        backbone.save_lora_weights(lora_path)
        ref_out = backbone(input_data)

        # Reload into a freshly constructed backbone. It is auto-named
        # differently, which exercises the stable path-based identification.
        reloaded = BertBackbone(**init_kwargs)
        reloaded.load_weights(base_path)
        reloaded.enable_lora(4)
        reloaded.load_lora_weights(lora_path)
        new_out = reloaded(input_data)

        self.assertAllClose(ref_out, new_out, rtol=1e-5, atol=1e-5)

    def test_lora_weights_use_stable_path_keys(self):
        init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        backbone = BertBackbone(**init_kwargs)
        backbone.enable_lora(4)
        lora_path = os.path.join(self.get_temp_dir(), "model.lora.h5")
        backbone.save_lora_weights(lora_path)

        # LoRA weights are keyed by the (stable, unique) layer path rather
        # than by an unstable integer index.
        with h5py.File(lora_path, "r") as f:
            top_level = set(f["lora"].keys())
        self.assertIn("transformer_layer_0", top_level)
        self.assertNotIn("0", top_level)

        # Backward compatibility: a file written with the legacy integer-index
        # keys must still load without error.
        legacy_path = os.path.join(self.get_temp_dir(), "legacy.lora.h5")
        store = keras.src.saving.saving_lib.H5IOStore(legacy_path, mode="w")
        lora_store = store.make("lora")
        lora_store["rank"] = backbone._lora_rank
        all_layers = [
            lyr
            for lyr in backbone._flatten_layers(include_self=False)
            if lyr.weights
        ]
        for layer_index in backbone._lora_enabled_layers:
            layer = all_layers[layer_index]
            inner_store = store.make(f"lora/{layer_index}")
            inner_store["lora_kernel_a"] = layer.lora_kernel_a
            inner_store["lora_kernel_b"] = layer.lora_kernel_b
        store.close()

        reloaded = BertBackbone(**init_kwargs)
        reloaded.enable_lora(4)
        reloaded.load_lora_weights(legacy_path)  # must not raise

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            Backbone.from_preset("bert_tiny_en_uncased", load_weights=False),
            BertBackbone,
        )
        self.assertIsInstance(
            Backbone.from_preset("gpt2_base_en", load_weights=False),
            GPT2Backbone,
        )

    @pytest.mark.large
    def test_from_preset_with_kwargs(self):
        # Test `dtype`
        backbone = Backbone.from_preset(
            "bert_tiny_en_uncased", load_weights=False, dtype="bfloat16"
        )
        self.assertIsInstance(backbone, BertBackbone)
        self.assertEqual(backbone.dtype_policy.name, "bfloat16")

        # Test kwargs forwarding
        backbone = Backbone.from_preset(
            "bert_tiny_en_uncased", load_weights=False, dropout=0.5
        )
        self.assertIsInstance(backbone, BertBackbone)
        self.assertAllClose(backbone.dropout, 0.5)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            GPT2Backbone.from_preset("bert_tiny_en_uncased", load_weights=False)
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            Backbone.from_preset("hf://spacy/en_core_web_sm")

    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        backbone = Backbone.from_preset("bert_tiny_en_uncased")
        backbone.save_to_preset(save_dir)

        # Check existence of files.
        self.assertTrue(os.path.exists(os.path.join(save_dir, CONFIG_FILE)))
        self.assertTrue(
            os.path.exists(os.path.join(save_dir, MODEL_WEIGHTS_FILE))
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, METADATA_FILE)))

        # Check the backbone config (`config.json`).
        backbone_config = load_json(save_dir, CONFIG_FILE)
        self.assertTrue("build_config" not in backbone_config)
        self.assertTrue("compile_config" not in backbone_config)

        # Check the metadata.
        metadata_config = load_json(save_dir, METADATA_FILE)
        self.assertTrue("keras_version" in metadata_config)
        self.assertTrue("keras_hub_version" in metadata_config)
        self.assertTrue("parameter_count" in metadata_config)
        self.assertTrue("TextClassifier" in metadata_config["tasks"])
        self.assertTrue("CausalLM" not in metadata_config["tasks"])

        # Try config class.
        self.assertEqual(BertBackbone, check_config_class(backbone_config))

        # Try loading the model from preset directory.
        restored_backbone = Backbone.from_preset(save_dir)

        data = {
            "token_ids": np.ones(shape=(2, 10), dtype="int32"),
            "segment_ids": np.array([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
            "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
        }

        # Check the model output.
        ref_out = backbone(data)
        new_out = restored_backbone(data)
        self.assertAllClose(ref_out, new_out)

    def test_export_supported_model(self):
        backbone_config = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 1,
            "hidden_dim": 512,
            "intermediate_dim": 1024,
            "head_dim": 128,
        }
        backbone = GemmaBackbone(**backbone_config)
        export_path = os.path.join(self.get_temp_dir(), "export_backbone")
        backbone.export_to_transformers(export_path)
        # Basic check: config file exists
        self.assertTrue(
            os.path.exists(os.path.join(export_path, "config.json"))
        )

    def test_export_unsupported_model(self):
        backbone_config = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 1,
            "hidden_dim": 512,
            "intermediate_dim": 1024,
            "head_dim": 128,
        }

        class UnsupportedBackbone(GemmaBackbone):
            pass

        backbone = UnsupportedBackbone(**backbone_config)
        export_path = os.path.join(self.get_temp_dir(), "unsupported")
        with self.assertRaises(ValueError):
            backbone.export_to_transformers(export_path)
