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
from keras import tree

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.samplers.sampler import Sampler


@keras_nlp_export("keras_nlp.samplers.BeamSampler")
class BeamSampler(Sampler):
    """Beam Sampler class.

    This sampler implements beam search algorithm. At each time-step, beam
    search keeps the beams (sequences) of the top `num_beams` highest
    accumulated probabilities, and uses each one of the beams to predict
    candidate next tokens.

    Args:
        num_beams: int. The number of beams that should be kept at each
            time-step. `num_beams` should be strictly positive.
        return_all_beams: bool. When set to `True`, the sampler will return all
            beams and their respective probabilities score.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    causal_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="beam")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_nlp.samplers.BeamSampler(num_beams=5)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        num_beams=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_beams = num_beams

    def start(self, data):
        batch_size = ops.shape(data["token_ids"])[0]
        data = tree.map_structure(self.create_beams, data)
        # Setup the initial beam log-likelihoods.
        log_probs = [[0.0] + [-1e9] * (self.num_beams - 1)]
        log_probs = ops.array(log_probs, dtype="float32")
        log_probs = self.flatten_beams(ops.repeat(log_probs, batch_size, 0))
        return {**data, "log_probabilities": log_probs}

    def next(
        self,
        data,
        index,
        logits,
    ):
        # Handle the case where logits lacks beams (during prefill).
        # In this case, we should add replicate the logits `num_beam` times.
        batch_size = ops.shape(data["token_ids"])[0] // self.num_beams
        if ops.shape(logits)[0] == batch_size:
            logits = self.create_beams(logits)

        probs = self.compute_probabilities(logits)
        log_probs = data["log_probabilities"]
        # Compute the running log-likelihood of each new candidate.
        next_log_probs = ops.log(probs) + log_probs[..., None]
        # Reshape `preds` to shape `(batch_size, num_beams * vocab_size)`.
        next_log_probs = ops.reshape(next_log_probs, [batch_size, -1])
        # Compute the top beam indices and next tokens.
        next_log_probs, indices = ops.top_k(
            next_log_probs, k=self.num_beams, sorted=False
        )
        vocab_size = ops.shape(logits)[-1]
        beam_indices = indices // vocab_size
        next_token = self.flatten_beams(indices % vocab_size)
        next_log_probs = self.flatten_beams(next_log_probs)
        # Work around for top_k output shape on tf backend.
        if keras.config.backend() == "tensorflow":
            # Work around for bug in top_k output shape on tf backend.
            import tensorflow as tf

            log_probs = tf.ensure_shape(next_log_probs, log_probs.shape)
        else:
            log_probs = next_log_probs

        def gather_beams(x):
            x = self.unflatten_beams(x)
            indices = beam_indices
            for axis in range(2, len(x.shape)):
                indices = ops.expand_dims(indices, axis=axis)
            x = ops.take_along_axis(x, indices, axis=1)
            return self.flatten_beams(x)

        data = tree.map_structure(gather_beams, data)
        next_index = index + 1
        token_ids, padding_mask = data["token_ids"], data["padding_mask"]
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
            "log_probabilities": log_probs,
        }

    def finish(
        self,
        data,
    ):
        data = tree.map_structure(self.unflatten_beams, data)
        top_beams = ops.argmax(data["log_probabilities"], axis=-1)

        def gather_beams(x):
            indices = top_beams
            for axis in range(1, len(x.shape)):
                indices = ops.expand_dims(indices, axis=axis)
            x = ops.take_along_axis(x, indices, axis=1)
            return self.flatten_beams(x)

        return tree.map_structure(gather_beams, data)

    def create_beams(self, x):
        return ops.repeat(x, self.num_beams, axis=0)

    def flatten_beams(self, x):
        return ops.reshape(x, (-1,) + ops.shape(x)[2:])

    def unflatten_beams(self, x):
        return ops.reshape(x, (-1, self.num_beams) + ops.shape(x)[1:])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_beams": self.num_beams,
            }
        )
        return config
