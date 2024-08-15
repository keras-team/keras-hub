# Copyright 2024 The KerasNLP Authors
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
from keras import ops
from keras import random

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.utils.tensor_utils import any_equal


@keras_nlp_export("keras_nlp.samplers.Sampler")
class Sampler:
    """Base sampler class.

    This base class can be extended to implement different auto-regressive
    sampling methods. To do so, override the `get_next_token()` method, which
    computes the next token based on a probability distribution over all
    possible vocab entries.

    Example:

    ```python
    causal_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Greedy search with some tokens forbidden.
    class CustomSampler(keras_nlp.samplers.Sampler):
        def __init__(self, forbidden_tokens, **kwargs):
            super().__init__(**kwargs)
            self.forbidden_tokens = forbidden_tokens

        def get_next_token(self, probs):
            batch_size, vocab_size = keras.ops.shape(probs)
            for id in self.forbidden_tokens:
                update = keras.ops.zeros((batch_size, 1))
                probs = keras.ops.slice_update(probs, (0, id), update)
            return keras.ops.argmax(probs, axis=-1)

    # 257 = "a" with a leading space, 262 = "the" with a leading space.
    causal_lm.compile(sampler=CustomSampler(forbidden_tokens=[257, 262]))
    causal_lm.summary()
    causal_lm.generate(["That's strange"])
    ```
    """

    def __init__(
        self,
        temperature=1.0,
    ):
        self.temperature = temperature
        self.generated_padding_id = 2
        self._seed_generators = []

    def __setattr__(self, name, value):
        # We could update to the `Tracker` class from keras-core if our needs
        # become more advanced (e.g. list assignment, nested trackables). For
        # now, we only track `SeedGenerator` instances directly on the sampler.
        if isinstance(value, random.SeedGenerator):
            self._seed_generators.append(value)
        return super().__setattr__(name, value)

    @property
    def variables(self):
        variables = []
        for sg in self._seed_generators:
            variables.append(sg.state)
        return variables

    def start(self, data):
        return data

    def has_next(
        self,
        data,
        index,
        stop_token_ids=None,
    ):
        # Check if we have reached `max_length`.
        token_ids, padding_mask = data["token_ids"], data["padding_mask"]
        _, max_length = ops.shape(token_ids)
        length_remaining = ops.less(index, max_length - 1)
        if stop_token_ids is None:
            return length_remaining
        # Check if all sequences have generated a *new* stop token.
        new_locations = ops.equal(padding_mask, self.generated_padding_id)
        new_end_tokens = any_equal(token_ids, stop_token_ids, new_locations)
        sequence_alive = ops.logical_not(ops.any(new_end_tokens, axis=-1))
        any_alive = ops.any(sequence_alive)
        return ops.logical_and(length_remaining, any_alive)

    def next(
        self,
        data,
        index,
        logits,
    ):
        next_index = index + 1
        token_ids, padding_mask = data["token_ids"], data["padding_mask"]
        # Compute the next token.
        probabilities = self.compute_probabilities(logits)
        next_token = self.get_next_token(probabilities)
        # Compute updated padding column.
        padding_column = padding_mask[:, next_index][:, None]
        next_padding = ops.ones_like(padding_column) * self.generated_padding_id
        next_padding = ops.where(padding_column, padding_column, next_padding)
        # Compute updated token id column.
        token_column = token_ids[:, next_index][:, None]
        next_token = ops.cast(next_token, token_ids.dtype)[:, None]
        next_token = ops.where(padding_column, token_column, next_token)
        # Update both in our data dictionary.
        start = [0, next_index]
        return {
            **data,
            "token_ids": ops.slice_update(token_ids, start, next_token),
            "padding_mask": ops.slice_update(padding_mask, start, next_padding),
        }

    def finish(self, data):
        return data

    def compute_probabilities(self, logits):
        """Compute token probabilities from logits.

        This will always be done in full precision, regardless of dtype, and
        scale by `temperature`.
        """
        logits = ops.cast(logits, "float32")
        return keras.activations.softmax(logits / self.temperature)

    def get_next_token(self, probabilities):
        """Get the next token.

        Args:
            probabilities: a Tensor, the probability distribution for next
                token over all vocab tokens.

        Get the next token based on given probability distribution over tokens.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"temperature": self.temperature}
