# Copyright 2022 The KerasNLP Authors
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
Small utility script to count parameters in our preset checkpoints.

Usage:
python tools/count_preset_params.py
"""

import inspect

from keras.utils.layer_utils import count_params
from tensorflow import keras

import keras_nlp

for name, symbol in keras_nlp.models.__dict__.items():
    if inspect.isclass(symbol) and issubclass(symbol, keras.Model):
        for preset in symbol.presets:
            model = symbol.from_preset(preset)
            params = count_params(model.weights)
            print(f"{name} {preset} {params}")
