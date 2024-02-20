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
"""ELECTRA model preset configurations."""

backbone_presets = {
    "electra_base_discriminator_en": {
        "metadata": {
            "description": ("ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"
                            "This is base discriminator model with 12 layers."
                            ),
            "params": "109482240",
            "official_name": "ELECTRA",
            "path": "electra",
            "model_card": "https://github.com/google-research/electra/blob/master/README.md"
        },
        "kaggle_handle": "kaggle://pranavprajapati16/electra/keras/electra_base_discriminator_en/2"
    }
}