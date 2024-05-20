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

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )
from keras import ops
from keras import tree

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.samplers.sampler import Sampler
from keras_nlp.src.utils.tensor_utils import any_equal


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
        return_all_beams=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_beams = num_beams
        self.return_all_beams = return_all_beams

    def __call__(
        self,
        next,
        prompt,
        cache=None,
        index=0,
        mask=None,
        stop_token_ids=None,
        hidden_states=None,
        model=None,
    ):
        batch_size, max_length = ops.shape(prompt)[0], ops.shape(prompt)[1]
        index = ops.cast(index, "int32")

        def create_beams(x):
            """Add initial beam state."""
            return ops.repeat(x, self.num_beams, axis=0)

        def flatten_beams(x):
            """Combine the beam dim and batch dim."""
            flat_shape = (batch_size * self.num_beams,) + ops.shape(x)[2:]
            return ops.reshape(x, flat_shape)

        def unflatten_beams(x):
            """Separate the beam dim and batch dim."""
            unflat_shape = (batch_size, self.num_beams) + ops.shape(x)[1:]
            return ops.reshape(x, unflat_shape)

        if mask is None:
            mask = ops.zeros_like(prompt, dtype="bool")
        else:
            mask = ops.cast(mask, dtype="bool")
        # `ops.while_loop` will not accept `None` as a value for `loop_vars`.
        has_cache = cache is not None
        cache = cache if has_cache else ()
        # Add extra sequences for each beam.
        prompt, mask = create_beams(prompt), create_beams(mask)
        cache = tree.map_structure(create_beams, cache)
        # Setup the initial beam log-likelihoods.
        # On the first loop, make sure only the original beam is considered.
        log_probs = ops.array(
            [[0.0] + [-1e9] * (self.num_beams - 1)], dtype="float32"
        )
        log_probs = flatten_beams(ops.repeat(log_probs, batch_size, axis=0))

        def cond(prompt, cache, index, log_probs):
            if stop_token_ids is None:
                return True
            # Stop if all sequences have produced a *new* stop token.
            end_tokens = any_equal(prompt, stop_token_ids, ~mask)
            prompt_done = ops.any(end_tokens, axis=-1)
            return ops.logical_not(ops.all(prompt_done))

        def body(prompt, cache, index, log_probs):
            # Compute the softmax distribution for the next token.
            logits, _, cache = next(prompt, cache, index)
            vocab_size = ops.shape(logits)[-1]
            probs = self.compute_probabilities(logits)

            # Compute the running log-likelihood of each new candidate.
            next_log_probs = ops.log(probs) + log_probs[..., None]
            # Reshape `preds` to shape `(batch_size, num_beams * vocab_size)`.
            next_log_probs = ops.reshape(next_log_probs, [batch_size, -1])

            # Compute the top beam indices and next tokens.
            next_log_probs, indices = ops.top_k(
                next_log_probs, k=self.num_beams, sorted=False
            )
            beam_indices = indices // vocab_size
            next_token = flatten_beams(indices % vocab_size)
            # We need `ensure_shape` as `top_k` will change the static shape.
            next_log_probs = flatten_beams(next_log_probs)
            # Work around for top_k output shape on tf backend.
            if isinstance(log_probs, tf.Tensor):
                log_probs = tf.ensure_shape(next_log_probs, log_probs.shape)
            else:
                log_probs = next_log_probs

            def gather_beams(x):
                x = unflatten_beams(x)
                indices = beam_indices
                for axis in range(2, len(x.shape)):
                    indices = ops.expand_dims(indices, axis=axis)
                x = ops.take_along_axis(x, indices, axis=1)
                return flatten_beams(x)

            prompt = gather_beams(prompt)
            if has_cache:
                cache = tree.map_structure(gather_beams, cache)

            # Update each beam with the next token.
            next_token = ops.cast(next_token, prompt.dtype)
            # Don't overwrite anywhere mask is True.
            next_token = ops.where(mask[:, index], prompt[:, index], next_token)
            # Update the prompt with the next token.
            next_token = next_token[:, None]
            prompt = ops.slice_update(prompt, [0, index], next_token)
            # Return the iteration of the loop state.
            return (prompt, cache, index + 1, log_probs)

        prompt, _, _, log_probs = self.run_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, cache, index, log_probs),
            maximum_iterations=(max_length - index),
            model=model,
        )

        all_prompts = unflatten_beams(prompt)
        all_log_probs = unflatten_beams(log_probs)

        if self.return_all_beams:
            sorted_indices = ops.argsort(-all_log_probs, axis=-1)
            sorted_log_probs = ops.take_along_axis(
                all_log_probs,
                sorted_indices,
                axis=1,
            )
            sorted_prompts = ops.take_along_axis(
                all_prompts,
                ops.expand_dims(sorted_indices, -1),
                axis=1,
            )
            return sorted_prompts, sorted_log_probs
        else:
            # Gather the top beam at each batch index.
            top_beams = ops.argmax(all_log_probs, axis=-1)[:, None, None]
            prompt = ops.take_along_axis(all_prompts, top_beams, axis=1)
            return ops.squeeze(prompt, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_beams": self.num_beams,
                "return_all_beams": self.return_all_beams,
            }
        )
        return config
