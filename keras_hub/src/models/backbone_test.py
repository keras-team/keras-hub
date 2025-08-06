import os

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
    def setUp(self):
        # Common config for backbone instantiation in export tests
        self.backbone_config = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 1,
            "hidden_dim": 512,
            "intermediate_dim": 1024,
            "head_dim": 128,
        }

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
        backbone = GemmaBackbone(**self.backbone_config)
        export_path = os.path.join(self.get_temp_dir(), "export_backbone")
        backbone.export_to_transformers(export_path)
        # Basic check: config file exists
        self.assertTrue(
            os.path.exists(os.path.join(export_path, "config.json"))
        )

    def test_export_unsupported_model(self):
        class UnsupportedBackbone(GemmaBackbone):
            pass

        backbone = UnsupportedBackbone(**self.backbone_config)
        export_path = os.path.join(self.get_temp_dir(), "unsupported")
        with self.assertRaises(ValueError):
            backbone.export_to_transformers(export_path)
