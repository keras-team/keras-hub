# Copyright 2023 The KerasNLP Authors
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

import datetime
import json
import os

from keras_nlp.backend import keras
from keras_nlp.model.bart import bart_presets
from keras_nlp.models import AlbertMaskedLM
from keras_nlp.models import BartSeq2SeqLM
from keras_nlp.models import BertMaskedLM
from keras_nlp.models.albert import albert_presets
from keras_nlp.models.bert import bert_presets
from keras_nlp.tests.test_case import TestCase
from keras_nlp.utils import preset_utils


class PresetUtilsTest(TestCase):
    def test_albert_sentencepiece_preset_saving(self):
        save_dir = self.get_temp_dir()
        model = AlbertMaskedLM.from_preset("albert_base_en_uncased")
        preset_utils.save_to_preset(model, save_dir)

        # Check existence of files
        self.assertTrue(
            os.path.exists(
                os.path.join(save_dir, "assets/tokenizer/vocabulary.txt")
            )
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, "config.json")))
        self.assertTrue(
            os.path.exists(os.path.join(save_dir, "model.weights.h5"))
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, "metadata.json")))

        preset_dict = albert_presets.backbone_presets["albert_base_en_uncased"]

        # Check that saved vocab is equal to the original preset vocab
        proto = open(
            os.path.join(save_dir, "assets/tokenizer/vocabulary.txt"), "rb"
        ).read()
        expected_proto_file = keras.utils.get_file(
            "vocab.spm",
            preset_dict["spm_proto_url"],
            cache_subdir=os.path.join("models", "albert_base_en_uncased"),
            file_hash=preset_dict["spm_proto_hash"],
        )
        expected_proto = open(expected_proto_file, "rb").read()
        self.assertEqual(proto, expected_proto)

        # Check the model config (`config.json``)
        config_json = open(os.path.join(save_dir, "config.json"), "r").read()
        self.assertTrue(
            "build_config" not in config_json
        )  # Test on raw json to include nested keys
        self.assertTrue(
            "compile_config" not in config_json
        )  # Test on raw json to include nested keys
        config = json.loads(config_json)
        self.assertAllEqual(
            config["assets"], ["assets/tokenizer/vocabulary.txt"]
        )
        self.assertEqual(config["weights"], "model.weights.h5")

        # Try deserializing the model using the config
        restored_model = keras.saving.deserialize_keras_object(config)
        restored_model.load_weights(os.path.join(save_dir, "model.weights.h5"))
        restored_model.preprocessor.tokenizer.load_assets(
            os.path.join(save_dir, "assets/tokenizer/")
        )

        # Check model outputs
        train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
        )
        input_data = model.preprocessor(*train_data)[0]
        self.assertAllEqual(model(input_data), restored_model(input_data))
