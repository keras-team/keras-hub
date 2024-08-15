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

from keras import ops
from keras import tree

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.samplers.sampler import Sampler


@keras_nlp_export("keras_nlp.samplers.ContrastiveSampler")
class ContrastiveSampler(Sampler):
    """Contrastive Sampler class.

    This sampler implements contrastive search algorithm. In short, the sampler
    chooses the token having the max "score" as the next token. The "score" is
    a weighted sum between token's probability and max similarity against
    previous tokens. By using this joint score, contrastive sampler reduces the
    behavior of duplicating seen tokens.

    Args:
        k: int, the `k` value of top-k. Next token will be chosen from k tokens.
        alpha: float, the weight of minus max similarity in joint score
            computation. The larger the value of `alpha`, the score relies more
            on the similarity than the token probability.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    causal_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="contrastive")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_nlp.samplers.ContrastiveSampler(k=5)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        k=5,
        alpha=0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.alpha = alpha

    def start(self, data):
        # We will treat contrastive search very similar to beam search, where we
        # explore k "beams" at any given time.
        batch_size = ops.shape(data["token_ids"])[0]
        data = tree.map_structure(self.create_beams, data)
        return {
            **data,
            "probabilities": ops.zeros((batch_size * self.k,)),
        }

    def has_next(self, data, index, end_token_id=None):
        # Allow sampling to go one extra index then normal to compute the hidden
        # states at the final index.
        return super().has_next(data, index - 1, end_token_id)

    def next(
        self,
        data,
        index,
        logits,
    ):
        probs, hidden_states = data["probabilities"], data["hidden_states"]
        batch_size, max_length = ops.shape(data["token_ids"])
        batch_size = batch_size // self.k

        # Handle the case where logits lacks beams (during prefill).
        # In this case, we should add replicate the logits `num_beam` times.
        if ops.shape(logits)[0] == batch_size:
            logits = self.create_beams(logits)

        # Compute the max similarity score for each top-k candidate.
        current_state = hidden_states[:, index, :]
        similarity_score = self.similarity(hidden_states, current_state)
        # Replace all future indices with -1, the lowest similarity score.
        score_mask = ops.expand_dims(ops.arange(max_length) < index, 0)
        similarity_score = ops.where(score_mask, similarity_score, -1)
        similarity_score = ops.max(similarity_score, axis=1)
        # Merge probabilities and similarities to a score for each candidate.
        score = (1 - self.alpha) * probs - self.alpha * similarity_score

        # For each original sequence, gather the best candidates by score.
        data = tree.map_structure(self.unflatten_beams, data)
        score = self.unflatten_beams(score)
        logits = self.unflatten_beams(logits)
        best_beam_indices = ops.argmax(score, axis=1)

        def get_best_beams(beams):
            indices = best_beam_indices
            for axis in range(1, len(beams.shape)):
                indices = ops.expand_dims(indices, axis=axis)
            best = ops.take_along_axis(beams, indices, axis=1)
            return ops.squeeze(best, axis=1)

        data = tree.map_structure(get_best_beams, data)
        logits = get_best_beams(logits)

        # Compute the softmax distribution the winning tokens.
        probs = self.compute_probabilities(logits)
        # Get new top-k candidate tokens and their probabilities.
        probs, next_token = ops.top_k(probs, k=self.k, sorted=False)
        probs = self.flatten_beams(probs)
        next_token = self.flatten_beams(next_token)

        data = tree.map_structure(self.create_beams, data)
        # Contrastive search runs one more iteration than usual, to compute the
        # the hidden_states at the final index. In this case, we need to be
        # careful to not update out of bounds tokens. We can simply clamp
        # `next_index` as our padding mask keeps us from overwriting tokens.
        next_index = ops.minimum(index + 1, max_length - 1)
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
            "probabilities": probs,
        }

    def finish(self, data):
        # We already gathered the top final tokens in the last iteration.
        return tree.map_structure(self.remove_beams, data)

    def similarity(self, h1, h2):
        h2 = ops.expand_dims(h2, -1)
        h1_norm = ops.sqrt(ops.sum(h1 * h1, axis=-1))
        h2_norm = ops.sqrt(ops.sum(h2 * h2, axis=-2))
        return ops.squeeze(ops.matmul(h1, h2), axis=-1) / (h1_norm * h2_norm)

    def create_beams(self, x):
        return ops.repeat(x, self.k, axis=0)

    def flatten_beams(self, x):
        return ops.reshape(x, (-1,) + ops.shape(x)[2:])

    def unflatten_beams(self, x):
        return ops.reshape(x, (-1, self.k) + ops.shape(x)[1:])

    def remove_beams(self, x):
        return self.unflatten_beams(x)[:, 0, ...]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "alpha": self.alpha,
            }
        )
        return config
