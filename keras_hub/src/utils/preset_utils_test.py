import json
import os

import keras
import pytest
from absl.testing import parameterized

from keras_hub.src.models.albert.albert_text_classifier import (
    AlbertTextClassifier,
)
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import sharded_weights_available
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.preset_utils import upload_preset


class PresetUtilsTest(TestCase):
    @pytest.mark.large
    def test_sharded_weights(self):
        if not sharded_weights_available():
            self.skipTest("Sharded weights are not available.")

        # Gemma2 config.
        init_kwargs = {
            "vocabulary_size": 4096,  # 256128
            "num_layers": 24,  # 46
            "num_query_heads": 16,  # 32
            "num_key_value_heads": 8,  # 16
            "hidden_dim": 64,  # 4608
            "intermediate_dim": 128,  # 73728
            "head_dim": 8,  # 128
            "sliding_window_size": 5,  # 4096
            "attention_logit_soft_cap": 50,
            "final_logit_soft_cap": 30,
            "layer_norm_epsilon": 1e-6,
            "query_head_dim_normalize": False,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "use_sliding_window_attention": True,
        }
        backbone = GemmaBackbone(**init_kwargs)  # ~4.4MB

        # Save the sharded weights.
        preset_dir = self.get_temp_dir()
        preset_saver = get_preset_saver(preset_dir)
        preset_saver.save_backbone(backbone, max_shard_size=0.002)
        self.assertTrue(
            os.path.exists(os.path.join(preset_dir, "model.weights.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(preset_dir, "model_00000.weights.h5"))
        )

        # Load the sharded weights.
        revived_backbone = GemmaBackbone.from_preset(preset_dir)
        for v1, v2 in zip(
            backbone.trainable_variables, revived_backbone.trainable_variables
        ):
            self.assertAllClose(v1, v2)

    @pytest.mark.large
    def test_preset_errors(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            AlbertTextClassifier.from_preset(AlbertTextClassifier)

        with self.assertRaisesRegex(ValueError, "Unknown preset identifier"):
            AlbertTextClassifier.from_preset("snaggle://bort/bort/bort")

        backbone = BertBackbone.from_preset("bert_tiny_en_uncased")
        preset_dir = self.get_temp_dir()
        config = keras.utils.serialize_keras_object(backbone)
        config["registered_name"] = "keras_hub>BortBackbone"
        with open(os.path.join(preset_dir, CONFIG_FILE), "w") as config_file:
            config_file.write(json.dumps(config, indent=4))
        with self.assertRaisesRegex(ValueError, "class keras_hub>BortBackbone"):
            BertBackbone.from_preset(preset_dir)

    @pytest.mark.large
    def test_tf_file_io(self):
        # Load a model from Kaggle to use as a test model.
        preset = "bert_tiny_en_uncased"
        backbone = BertBackbone.from_preset(preset)
        # Save the model on a local directory.
        temp_dir = self.get_temp_dir()
        local_preset_dir = os.path.join(temp_dir, "bert_preset")
        backbone.save_to_preset(local_preset_dir)
        # Load with "file://" which tf supports.
        backbone = BertBackbone.from_preset("file://" + local_preset_dir)

    @pytest.mark.large
    def test_upload_empty_preset(self):
        temp_dir = self.get_temp_dir()
        empty_preset = os.path.join(temp_dir, "empty")
        os.mkdir(empty_preset)
        uri = "kaggle://test/test/test"

        with self.assertRaises(FileNotFoundError):
            upload_preset(uri, empty_preset)

    @parameterized.parameters((CONFIG_FILE), ("model.weights.h5"))
    @pytest.mark.large
    def test_upload_with_missing_file(self, missing_file):
        # Load a model from Kaggle to use as a test model.
        preset = "bert_tiny_en_uncased"
        backbone = BertBackbone.from_preset(preset)
        tokenizer = BertTokenizer.from_preset(preset)

        # Save the model on a local directory.
        temp_dir = self.get_temp_dir()
        local_preset_dir = os.path.join(temp_dir, "bert_preset")
        backbone.save_to_preset(local_preset_dir)
        tokenizer.save_to_preset(local_preset_dir)

        # Delete the file that is supposed to be missing.
        missing_path = os.path.join(local_preset_dir, missing_file)
        os.remove(missing_path)

        # Verify error handling.
        with self.assertRaisesRegex(FileNotFoundError, "is missing"):
            upload_preset("kaggle://user/model/keras/variant", local_preset_dir)

    @pytest.mark.large
    def test_upload_with_invalid_json(self):
        # Load a model from Kaggle to use as a test model.
        preset = "bert_tiny_en_uncased"
        backbone = BertBackbone.from_preset(preset)
        tokenizer = BertTokenizer.from_preset(preset)

        # Save the model on a local directory.
        temp_dir = self.get_temp_dir()
        local_preset_dir = os.path.join(temp_dir, "bert_preset")
        backbone.save_to_preset(local_preset_dir)
        tokenizer.save_to_preset(local_preset_dir)

        # Re-write json file content to an invalid format.
        json_path = os.path.join(local_preset_dir, CONFIG_FILE)
        with open(json_path, "w") as file:
            file.write("Invalid!")

        # Verify error handling.
        with self.assertRaisesRegex(ValueError, "is an invalid json"):
            upload_preset("kaggle://test/test/test", local_preset_dir)
