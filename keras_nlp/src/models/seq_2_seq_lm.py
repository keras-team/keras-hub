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

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.causal_lm import CausalLM


@keras_nlp_export("keras_nlp.models.Seq2SeqLM")
class Seq2SeqLM(CausalLM):
    """Base class for sequence to sequence language modeling tasks.

    `Seq2SeqLM` tasks wrap a `keras_nlp.models.Backbone` and
    a `keras_nlp.models.Preprocessor` to create a model that can be used for
    generation and generative fine-tuning, when generation is conditioned on
    additional input sequence in a sequence-to-sequence setting.

    `Seq2SeqLM` tasks provide an additional, high-level `generate()` function
    which can be used to auto-regressively sample an output sequence token by
    token. The `compile()` method of `Seq2SeqLM` classes contains an additional
    `sampler` argument, which can be used to pass a `keras_nlp.samplers.Sampler`
    to control how the predicted distribution will be sampled.

    When calling `fit()`, each input should contain an input and output
    sequence. The model will be trained to predict the output sequence
    token-by-token using a causal mask, similar to a `keras_nlp.models.CausalLM`
    task. Unlike the `CausalLM` task, an input sequence must be passed, and
    can be attended to in full by all tokens in the output sequence.

    All `Seq2SeqLM` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.

    Example:
    ```python
    # Load a Bart backbone with pre-trained weights.
    seq_2_seq_lm = keras_nlp.models.Seq2SeqLM.from_preset(
        "bart_base_en",
    )
    seq_2_seq_lm.compile(sampler="top_k")
    # Generate conditioned on the `"The quick brown fox."` as an input sequence.
    seq_2_seq_lm.generate("The quick brown fox.", max_length=30)
    ```
    """

    def build_cache(self, batch_size, encoder_max_length, decoder_max_length):
        raise NotImplementedError

    def compute_cross_attention_cache(
        self, encoder_token_ids, encoder_padding_mask
    ):
        raise NotImplementedError

    def call_with_cache(
        self,
        token_ids,
        cache,
        index,
        encoder_padding_mask,
    ):
        raise NotImplementedError

    def prefill(self, data):
        """Run inference on the entire input sequence to seed generate data."""
        batch_size, max_length = ops.shape(data["decoder_token_ids"])
        cache = self.build_cache(batch_size, max_length)
        cross_attention_cache = self.compute_cross_attention_cache(
            encoder_token_ids=data["encoder_token_ids"],
            encoder_padding_mask=data["encoder_padding_mask"],
        )
        # Run a forward pass with the full padded token id sequence.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=data["decoder_token_ids"],
            cache=cache,
            index=0,
            cross_attention_cache=cross_attention_cache,
            encoder_padding_mask=data["encoder_padding_mask"],
        )
        # Sampling data.
        data = {
            "token_ids": data["decoder_token_ids"],
            "padding_mask": data["decoder_padding_mask"],
            "cache": cache,
            "hidden_states": hidden_states,
            # Extra data for seq2seq decoding.
            "encoder_token_ids": data["encoder_token_ids"],
            "encoder_padding_mask": data["encoder_padding_mask"],
            "cross_attention_cache": cross_attention_cache,
        }
        # Add sampling beams, other sampling state.
        data = self.sampler.start(data)
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(data["padding_mask"], axis=-1)
        # Start at the last index that has all user inputted ids.
        index = ops.min(row_lengths) - 1
        # Generate one token from the logits we just computed.
        data = self.sampler.next(
            data=data,
            index=index,
            logits=logits[:, index, :],
        )
        return data, index + 1

    def is_decoding(self, data, index, end_token_id=None):
        return self.sampler.has_next(
            data=data,
            index=index,
            end_token_id=end_token_id,
        )

    def decode(self, data, index):
        # Run a forward pass with a single token id, and full length cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=data["token_ids"][:, index][:, None],
            cache=data["cache"],
            index=index,
            cross_attention_cache=data["cross_attention_cache"],
            encoder_padding_mask=data["encoder_padding_mask"],
        )
        # Update our data dict.
        data = {
            **data,
            "cache": cache,
            "hidden_states": ops.slice_update(
                data["hidden_states"], [0, index, 0], hidden_states
            ),
        }
        # Generate the next token.
        data = self.sampler.next(
            data=data,
            index=index,
            logits=logits[:, 0, :],
        )
        return data, index + 1

    def finish_decoding(self, data):
        data = self.sampler.finish(data)
        return {
            "decoder_token_ids": data["token_ids"],
            "decoder_padding_mask": data["padding_mask"],
            "encoder_token_ids": data["encoder_token_ids"],
            "encoder_padding_mask": data["encoder_padding_mask"],
        }
