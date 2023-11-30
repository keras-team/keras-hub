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
"""
This script was used to convert our legacy presets into the directory format
used by Kaggle.

This script is for reference only.
"""

import os
import shutil

os.environ["KERAS_HOME"] = os.getcwd()

import keras_nlp  # noqa: E402
from keras_nlp.src.utils.preset_utils import save_to_preset  # noqa: E402

BUCKET = "keras-nlp-kaggle"

backbone_models = [
    (keras_nlp.models.AlbertBackbone, keras_nlp.models.AlbertTokenizer),
    (keras_nlp.models.BartBackbone, keras_nlp.models.BartTokenizer),
    (keras_nlp.models.BertBackbone, keras_nlp.models.BertTokenizer),
    (keras_nlp.models.DebertaV3Backbone, keras_nlp.models.DebertaV3Tokenizer),
    (keras_nlp.models.DistilBertBackbone, keras_nlp.models.DistilBertTokenizer),
    (keras_nlp.models.FNetBackbone, keras_nlp.models.FNetTokenizer),
    (keras_nlp.models.GPT2Backbone, keras_nlp.models.GPT2Tokenizer),
    (keras_nlp.models.OPTBackbone, keras_nlp.models.OPTTokenizer),
    (keras_nlp.models.RobertaBackbone, keras_nlp.models.RobertaTokenizer),
    (keras_nlp.models.T5Backbone, keras_nlp.models.T5Tokenizer),
    (keras_nlp.models.WhisperBackbone, keras_nlp.models.WhisperTokenizer),
    (keras_nlp.models.XLMRobertaBackbone, keras_nlp.models.XLMRobertaTokenizer),
]
for backbone_cls, tokenizer_cls in backbone_models:
    for preset in backbone_cls.presets:
        backbone = backbone_cls.from_preset(preset)
        tokenizer = tokenizer_cls.from_preset(preset)
        save_to_preset(
            backbone,
            preset,
            config_filename="config.json",
        )
        save_to_preset(
            tokenizer,
            preset,
            config_filename="tokenizer.json",
        )
        # Delete first to clean up any exising version.
        os.system(f"gsutil rm -rf gs://{BUCKET}/{preset}")
        os.system(f"gsutil cp -r {preset} gs://{BUCKET}/{preset}")
        for root, _, files in os.walk(preset):
            for file in files:
                path = os.path.join(BUCKET, root, file)
                os.system(
                    f"gcloud storage objects update gs://{path} "
                    "--add-acl-grant=entity=AllUsers,role=READER"
                )
        # Clean up local disk usage.
        shutil.rmtree("models")
        shutil.rmtree(preset)

# Handle our single task model.
preset = "bert_tiny_en_uncased_sst2"
task = keras_nlp.models.BertClassifier.from_preset(preset)
tokenizer = keras_nlp.models.BertTokenizer.from_preset(preset)
save_to_preset(
    task,
    preset,
    config_filename="config.json",
)
save_to_preset(
    tokenizer,
    preset,
    config_filename="tokenizer.json",
)
# Delete first to clean up any exising version.
os.system(f"gsutil rm -rf gs://{BUCKET}/{preset}")
os.system(f"gsutil cp -r {preset} gs://{BUCKET}/{preset}")
for root, _, files in os.walk(preset):
    for file in files:
        path = os.path.join(BUCKET, root, file)
        os.system(
            f"gcloud storage objects update gs://{path} "
            "--add-acl-grant=entity=AllUsers,role=READER"
        )
# Clean up local disk usage.
shutil.rmtree("models")
shutil.rmtree(preset)
