# Copyright 2024 The KerasHub Authors
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
import keras

from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.models.stable_diffusion_v3.clip_tokenizer import (
    CLIPTokenizer,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


class CLIPPreprocessor(Preprocessor):
    tokenizer_cls = CLIPTokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=77,
        add_start_token=True,
        add_end_token=False,
        to_lower=True,
        pad_with_end_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.to_lower = to_lower
        self.pad_with_end_token = pad_with_end_token

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        pad_value = self.tokenizer.pad_token_id
        if self.pad_with_end_token:
            pad_value = self.tokenizer.end_token_id

        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=pad_value,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if self.to_lower:
            x = tf.strings.lower(x)
        token_ids, padding_mask = self.packer(
            self.tokenizer(x),
            sequence_length=sequence_length or self.sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
                "to_lower": self.to_lower,
                "pad_with_end_token": self.pad_with_end_token,
            }
        )
        return config
