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
import re
import shutil

os.environ["KERAS_HOME"] = os.getcwd()

from keras_nlp import models  # noqa: E402
from keras_nlp.src.utils.preset_utils import save_to_preset  # noqa: E402

BUCKET = "keras-nlp-kaggle"


def to_snake_case(name):
    name = re.sub(r"\W+", "", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
    return name


if __name__ == "__main__":
    backbone_models = [
        (models.AlbertBackbone, models.AlbertTokenizer),
        (models.BartBackbone, models.BartTokenizer),
        (models.BertBackbone, models.BertTokenizer),
        (models.DebertaV3Backbone, models.DebertaV3Tokenizer),
        (models.DistilBertBackbone, models.DistilBertTokenizer),
        (models.FNetBackbone, models.FNetTokenizer),
        (models.GPT2Backbone, models.GPT2Tokenizer),
        (models.OPTBackbone, models.OPTTokenizer),
        (models.RobertaBackbone, models.RobertaTokenizer),
        (models.T5Backbone, models.T5Tokenizer),
        (models.WhisperBackbone, models.WhisperTokenizer),
        (models.XLMRobertaBackbone, models.XLMRobertaTokenizer),
    ]
    for backbone_cls, tokenizer_cls in backbone_models:
        for preset in backbone_cls.presets:
            backbone = backbone_cls.from_preset(
                preset, name=to_snake_case(backbone_cls.__name__)
            )
            tokenizer = tokenizer_cls.from_preset(
                preset, name=to_snake_case(tokenizer_cls.__name__)
            )
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
    task = models.BertClassifier.from_preset(
        preset, name=to_snake_case(models.BertClassifier.__name__)
    )
    tokenizer = models.BertTokenizer.from_preset(
        preset, name=to_snake_case(models.BertTokenizer.__name__)
    )
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
