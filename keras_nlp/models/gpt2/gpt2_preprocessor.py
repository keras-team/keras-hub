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

"""GPT2 preprocessor layer."""

import copy

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty


class GPT2Preprocessor(keras.layers.Layer):
    def __init__(self, tokenizer, sequence_length, **kwargs):

        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def call(self, x, y=None, sample_weight=None):
        token_ids = self.tokenizer(x)
        mask = tf.ones_like(token_ids, dtype=tf.bool)
        mask = mask.to_tensor(shape=(None, self.sequence_length))
        token_ids = token_ids.to_tensor(shape=(None, self.sequence_length))
        x = {
            "token_ids": token_ids,
            "padding_mask": mask,
        }

        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        sequence_length=None,
        **kwargs,
    ):
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        tokenizer = GPT2Tokenizer.from_preset(preset)

        # Use model's `max_sequence_length` if `sequence_length` unspecified;
        # otherwise check that `sequence_length` not too long.
        metadata = cls.presets[preset]
        max_sequence_length = metadata["config"]["max_sequence_length"]
        if sequence_length is not None:
            if sequence_length > max_sequence_length:
                raise ValueError(
                    f"`sequence_length` cannot be longer than `{preset}` "
                    f"preset's `max_sequence_length` of {max_sequence_length}. "
                    f"Received: {sequence_length}."
                )
        else:
            sequence_length = max_sequence_length

        return cls(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            **kwargs,
        )


class GPT2CausalLMPreprocessor(GPT2Preprocessor):
    def call(self, x, y=None, sample_weight=None):
        token_ids = self.tokenizer(x)
        mask = tf.ones_like(token_ids, dtype=tf.bool)
        mask = mask.to_tensor(shape=(None, self.sequence_length))
        token_ids = token_ids.to_tensor(shape=(None, self.sequence_length))
        x = {
            "token_ids": token_ids[:, :-1],
            "padding_mask": mask[:, 1:],
        }

        y = token_ids[:, 1:]
        sample_weight = mask[:, 1:]

        return pack_x_y_sample_weight(x, y, sample_weight)
