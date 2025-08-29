import keras
from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.samplers.Sampler")
class Sampler:
    """Base sampler class.

    Args:
        temperature: float. optional. Used to control the
            randomness of the sampling. The higher the temperature, the
            more diverse the samples. Defaults to `1.0`.

    Call arguments:
        {{call_args}}

    This base class can be extended to implement different auto-regressive
    sampling methods. To do so, override the `get_next_token()` method, which
    computes the next token based on a probability distribution over all
    possible vocab entries.

    Example:

    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Greedy search with some tokens forbidden.
    class CustomSampler(keras_hub.samplers.Sampler):
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

    def __init__(self, temperature=1.0):
        self.temperature = temperature
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
        max_length = ops.shape(prompt)[-1]
        # Make sure `max_length` and `index` are the same dtype.
        index = ops.cast(index, "int32")
        max_length = ops.cast(max_length, "int32")
        batch_size = ops.shape(prompt)[0]
        if mask is None:
            mask = ops.zeros_like(prompt, dtype="bool")
        else:
            mask = ops.cast(mask, dtype="bool")
        # `ops.while_loop` will not accept `None` as a value for `loop_vars`.
        cache = () if cache is None else cache
        finished = ops.zeros([batch_size], dtype="bool")
        if stop_token_ids is not None:
            stop_token_ids_tensor = ops.convert_to_tensor(
                stop_token_ids, dtype=prompt.dtype
            )
        else:
            stop_token_ids_tensor = None

        # Compute generated_mask
        seq_length = ops.shape(prompt)[1]
        row_lengths = ops.sum(ops.cast(mask, "int32"), axis=-1)
        indices = ops.arange(seq_length, dtype="int32")
        indices = ops.expand_dims(indices, axis=0)
        generated_mask = indices >= ops.expand_dims(row_lengths, axis=-1)
        generated_mask = ops.cast(generated_mask, "bool")

        def cond(prompt, cache, index, finished):
            if stop_token_ids is None:
                return index < max_length
            return ops.logical_not(ops.all(finished))

        def body(prompt, cache, index, finished):
            logits, _, cache = next(prompt, cache, index)
            probabilities = self.compute_probabilities(logits)
            next_token = self.get_next_token(probabilities)
            next_token = ops.cast(next_token, prompt.dtype)
            # Preserve prompt tokens
            next_token = ops.where(mask[:, index], prompt[:, index], next_token)
            if stop_token_ids is not None:
                # Check stop tokens only for generated positions
                # and non-finished sequences
                is_generating = generated_mask[:, index] & ~finished
                is_stop = is_generating & ops.any(
                    next_token[:, None] == stop_token_ids_tensor, axis=-1
                )
                finished = ops.logical_or(finished, is_stop)
            next_token = next_token[:, None]
            prompt = ops.slice_update(prompt, [0, index], next_token)
            return (prompt, cache, index + 1, finished)

        prompt, _, _, _ = self.run_loop(
            cond,
            body,
            loop_vars=(prompt, cache, index, finished),
            maximum_iterations=(max_length - index),
            model=model,
        )
        return prompt

    def compute_probabilities(self, logits):
        """Compute token probabilities from logits.
         This will always be done in full precision, regardless of dtype, and
        scale by `temperature`.
        """
        logits = ops.cast(logits, "float32")
        return keras.activations.softmax(logits / self.temperature)

    def run_loop(
        self, cond, body, model=None, loop_vars=None, maximum_iterations=None
    ):
        if keras.config.backend() == "jax":
            import itertools

            if model:
                model_trainable_variables = model.trainable_variables
                model_non_trainable_variables = model.non_trainable_variables
            else:
                model_trainable_variables = []
                model_non_trainable_variables = []

            def stateless_cond(state, *loop_vars):
                return cond(*loop_vars)

            def stateless_body(state, *loop_vars):
                (
                    sampler_variables,
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                mapping = itertools.chain(
                    zip(self.variables, sampler_variables),
                    zip(model_trainable_variables, trainable_variables),
                    zip(model_non_trainable_variables, non_trainable_variables),
                )
                with keras.StatelessScope(state_mapping=mapping) as scope:
                    loop_vars = body(*loop_vars)
                    sampler_variables = []
                    for v in self.variables:
                        new_v = scope.get_current_value(v)
                        sampler_variables.append(
                            new_v if new_v is not None else v
                        )
                    state = (
                        sampler_variables,
                        trainable_variables,
                        non_trainable_variables,
                    )
                return state, *loop_vars

            variables = [ops.convert_to_tensor(v) for v in self.variables]
            trainable_variables = [
                ops.convert_to_tensor(v) for v in model_trainable_variables
            ]
            non_trainable_variables = [
                ops.convert_to_tensor(v) for v in model_non_trainable_variables
            ]
            state = (variables, trainable_variables, non_trainable_variables)
            state, *loop_vars = ops.while_loop(
                cond=stateless_cond,
                body=stateless_body,
                loop_vars=(state, *loop_vars),
                maximum_iterations=maximum_iterations,
            )
            for ref_v, v in zip(self.variables, state[0]):
                ref_v.assign(v)
        else:
            loop_vars = ops.while_loop(
                cond=cond,
                body=body,
                loop_vars=(loop_vars),
                maximum_iterations=maximum_iterations,
            )
        return loop_vars

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
