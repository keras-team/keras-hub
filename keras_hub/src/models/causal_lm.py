import itertools
from functools import partial

import keras
from keras import ops
from keras import tree

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task
from keras_hub.src.samplers.serialization import get as get_sampler
from keras_hub.src.utils.tensor_utils import any_equal

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.CausalLM")
class CausalLM(Task):
    """Base class for generative language modeling tasks.

    `CausalLM` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    generation and generative fine-tuning.

    `CausalLM` tasks provide an additional, high-level `generate()` function
    which can be used to auto-regressively sample a model token by token with a
    string in, string out signature. The `compile()` method of all `CausalLM`
    classes contains an additional `sampler` argument, which can be used to pass
    a `keras_hub.samplers.Sampler` to control how the predicted distribution
    will be sampled.

    When calling `fit()`, the tokenized input will be predicted token-by-token
    with a causal mask applied, which gives both a pre-training and supervised
    fine-tuning setup for controlling inference-time generation.

    All `CausalLM` tasks include a `from_preset()` constructor which can be used
    to load a pre-trained config and weights.

    Example:
    ```python
    # Load a GPT2 backbone with pre-trained weights.
    causal_lm = keras_hub.models.CausalLM.from_preset(
        "gpt2_base_en",
    )
    causal_lm.compile(sampler="top_k")
    causal_lm.generate("Keras is a", max_length=64)

    # Load a Mistral instruction tuned checkpoint at bfloat16 precision.
    causal_lm = keras_hub.models.CausalLM.from_preset(
        "mistral_instruct_7b_en",
        dtype="bfloat16",
    )
    causal_lm.compile(sampler="greedy")
    causal_lm.generate("Keras is a", max_length=64)
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="top_k",
        **kwargs,
    ):
        """Configures the `CausalLM` task for training and generation.

        The `CausalLM` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `weighted_metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        The `CausalLM` task adds a new `sampler` to `compile`, which can be used
        to control the sampling strategy used with the `generate` function.

        Note that because training inputs include padded tokens which are
        excluded from the loss, it is almost always a good idea to compile with
        `weighted_metrics` and not `metrics`.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.SparseCategoricalCrossentropy` loss will be
                applied for the token classification `CausalLM` task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            weighted_metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.SparseCategoricalAccuracy` will be
                applied to track the accuracy of the model at guessing masked
                token values. See `keras.Model.compile` and `keras.metrics` for
                more info on possible `weighted_metrics` values.
            sampler: A sampler name, or a `keras_hub.samplers.Sampler` instance.
                Configures the sampling method used during `generate()` calls.
                See `keras_hub.samplers` for a full list of built-in sampling
                strategies.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(2e-5)
        if loss == "auto":
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if weighted_metrics == "auto":
            weighted_metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            **kwargs,
        )
        self.sampler = get_sampler(sampler)
        # Clear the compiled generate function.
        self.generate_function = None

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        """A compilable generation function for a single batch of inputs.

        This default implementation works for all CausalLM models that
        implement `call_with_cache()` and `_build_cache()`. It includes
        backend-specific optimizations (e.g., direct tensor indexing on
        torch) that benefit all models automatically.

        Subclasses only need to override this if they require custom
        generation logic beyond what `call_with_cache` provides.

        Args:
            inputs: A dictionary with two keys `"token_ids"` and
                `"padding_mask"` and batched tensor values.
            stop_token_ids: List of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(token_ids)
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        # Use direct tensor indexing on torch backend to avoid
        # ops.slice / convert_to_tensor overhead. JAX/TF need ops.slice
        # for static shapes in JIT compilation.
        _use_direct_indexing = keras.config.backend() == "torch"

        if _use_direct_indexing:
            import torch

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            # Extract single token for cached forward pass.
            if _use_direct_indexing:
                prompt = prompt[:, cache_update_index : cache_update_index + 1]
                # Ensure cache_update_index is a tensor for
                # call_with_cache, as some models pass it through to
                # sublayers which require tensor-typed kwargs.
                cache_update_index = torch.tensor(
                    cache_update_index, dtype=torch.int32
                )
            else:
                batch_size = ops.shape(prompt)[0]
                prompt = ops.slice(
                    prompt, [0, cache_update_index], [batch_size, 1]
                )
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index,
            )
            return (
                logits[:, 0, :],
                hidden_states[:, 0, :],
                cache,
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            # Build a mask of stop tokens locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
    ):
        """Forward pass with cache for autoregressive inference.

        Subclasses must override this method to define their specific
        cached forward pass logic.

        Args:
            token_ids: a dense int Tensor with shape
                `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current
                inputs in the whole sequence.

        Returns:
            A (logits, hidden_states, cache) tuple.
        """
        raise NotImplementedError

    def _build_cache(self, token_ids):
        """Build an empty cache for use with `call_with_cache()`.

        Subclasses must override this method to define their specific
        cache structure.
        """
        raise NotImplementedError

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        self.generate_function = self.generate_step
        if keras.config.backend() == "openvino":
            from keras_hub.src.utils.openvino_utils import ov_infer

            def wrapped_generate_function(inputs, stop_token_ids=None):
                # Convert to numpy for OpenVINO backend
                inputs = tree.map_structure(ops.array, inputs)
                return ov_infer(
                    self, inputs, stop_token_ids, self.generate_step
                )

            self.generate_function = wrapped_generate_function
        if keras.config.backend() == "torch":
            import torch

            def wrapped_generate_function(
                inputs,
                stop_token_ids=None,
            ):
                # Use torch.no_grad() and inference_mode for best performance
                with torch.no_grad(), torch.inference_mode():
                    return self.generate_step(inputs, stop_token_ids)

            self.generate_function = wrapped_generate_function
        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            # `jit_compile` is a property of keras.Model after TF 2.12.
            # Use `getattr()` for backwards compatibility.
            jit_compile = getattr(self, "jit_compile", True)
            self.generate_function = tf.function(
                self.generate_step, jit_compile=jit_compile
            )
        elif keras.config.backend() == "jax" and not self.run_eagerly:
            import jax

            @partial(jax.jit, static_argnames=["stop_token_ids"])
            def compiled_generate_function(inputs, stop_token_ids, state):
                (
                    sampler_variables,
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                mapping = itertools.chain(
                    zip(self.sampler.variables, sampler_variables),
                    zip(self.trainable_variables, trainable_variables),
                    zip(self.non_trainable_variables, non_trainable_variables),
                )

                with keras.StatelessScope(state_mapping=mapping) as scope:
                    outputs = self.generate_step(inputs, stop_token_ids)

                # Get updated sampler variables from the stateless scope.
                sampler_variables = []
                for v in self.sampler.variables:
                    new_v = scope.get_current_value(v)
                    sampler_variables.append(new_v if new_v is not None else v)
                return outputs, sampler_variables

            def wrapped_generate_function(
                inputs,
                stop_token_ids=None,
            ):
                if isinstance(stop_token_ids, list):
                    stop_token_ids = tuple(stop_token_ids)

                # Create an explicit tuple of all variable state.
                state = (
                    [v.value for v in self.sampler.variables],
                    # Use the explicit variable.value to preserve the
                    # sharding spec of distribution.
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                inputs = tree.map_structure(ops.convert_to_tensor, inputs)
                outputs, sampler_variables = compiled_generate_function(
                    inputs,
                    stop_token_ids,
                    state,
                )
                # Only assign the sampler variables (random seeds), as other
                # model variables should never be updated in generation.
                for ref_v, v in zip(self.sampler.variables, sampler_variables):
                    ref_v.assign(v)
                return outputs

            self.generate_function = wrapped_generate_function

        return self.generate_function

    def _normalize_generate_inputs(
        self,
        inputs,
    ):
        """Normalize user input to the generate function.

        This function converts all inputs to tensors, adds a batch dimension if
        necessary, and returns a iterable "dataset like" object (either an
        actual `tf.data.Dataset` or a list with a single batch element).
        """
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        if self.preprocessor is None:
            return [inputs], False

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        if isinstance(inputs, dict):
            for key in inputs:
                inputs[key], input_is_scalar = normalize(inputs[key])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def _normalize_generate_outputs(
        self,
        outputs,
        input_is_scalar,
    ):
        """Normalize user output from the generate function.

        This function converts all output to numpy (for integer output), or
        python strings (for string output). If a batch dimension was added to
        the input, it is removed from the output (so generate can be string in,
        string out).
        """

        def normalize(x):
            if isinstance(x[0], list):
                outputs = []
                for batch in x:
                    for e in batch:
                        outputs.append(e)
                return outputs[0] if input_is_scalar else outputs
            outputs = ops.concatenate(x, axis=0)
            outputs = ops.squeeze(outputs, 0) if input_is_scalar else outputs
            return ops.convert_to_numpy(outputs)

        if isinstance(outputs[0], dict):
            normalized = {}
            for key in outputs[0]:
                normalized[key] = normalize([x[key] for x in outputs])
            return normalized
        return normalize([x for x in outputs])

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
    ):
        """Generate text given prompt `inputs`.

        This method generates text based on given `inputs`. The sampling method
        used for generation can be set via the `compile()` method.

        If `inputs` are a `tf.data.Dataset`, outputs will be generated
        "batch-by-batch" and concatenated. Otherwise, all inputs will be handled
        as a single batch.

        If a `preprocessor` is attached to the model, `inputs` will be
        preprocessed inside the `generate()` function and should match the
        structure expected by the `preprocessor` layer (usually raw strings).
        If a `preprocessor` is not attached, inputs should match the structure
        expected by the `backbone`. See the example usage above for a
        demonstration of each.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`. If a
                `preprocessor` is attached to the model, `inputs` should match
                the structure expected by the `preprocessor` layer. If a
                `preprocessor` is not attached, `inputs` should match the
                structure expected the `backbone` model.
            max_length: Optional. int. The max length of the generated sequence.
                Will default to the max configured `sequence_length` of the
                `preprocessor`. If `preprocessor` is `None`, `inputs` should be
                should be padded to the desired maximum length and this argument
                will be ignored.
            stop_token_ids: Optional. `None`, "auto", or tuple of token ids.
                Defaults to "auto" which uses the
                `preprocessor.tokenizer.end_token_id`. Not specifying a
                processor will produce an error. None stops generation after
                generating `max_length` tokens. You may also specify a list of
                token id's the model should stop on. Note that sequences of
                tokens will each be interpreted as a stop token, multi-token
                stop sequences are not supported.
            strip_prompt: Optional. By default, generate() returns the full
                prompt followed by its completion generated by the model. If
                this option is set to True, only the newly generated text is
                returned.
        """
        # Setup our three main passes.
        # 1. Optionally preprocessing strings to dense integer tensors.
        # 2. Generate new tokens via a compiled function on dense tensors.
        # 3. Optionally postprocess dense integer tensors back to string.
        generate_function = self.make_generate_function()

        if self.preprocessor is None and stop_token_ids == "auto":
            raise ValueError(
                "A `preprocessor` must be attached to the model if "
                '`stop_token_ids="auto"`. Currently `preprocessor=None`. To '
                "call `generate()` with preprocessing detached, either pass "
                "`stop_token_ids=None` to always generate until `max_length` "
                "or pass a tuple of token ids that should terminate generation "
                "as `stop_token_ids`."
            )
        elif stop_token_ids == "auto":
            stop_token_ids = [self.preprocessor.tokenizer.end_token_id]
            # Some models like Llama3 use two end tokens: <|eot_id|> in
            # "instruct" versions and <|end_of_text|> in others.
            if hasattr(self.preprocessor.tokenizer, "end_token2_id"):
                stop_token_ids.append(self.preprocessor.tokenizer.end_token2_id)

        def preprocess(x):
            return self.preprocessor.generate_preprocess(
                x, sequence_length=max_length
            )

        def generate(x):
            return generate_function(x, stop_token_ids=stop_token_ids)

        def strip_prompt_function(x, prompt):
            # This function removes the prompt from the generated
            # response, in a batch-friendly fashion.
            y = {}
            prompt_mask = prompt["padding_mask"]
            seq_len = prompt_mask.shape[1]

            # We need to shift every output sequence by the size of the prompt.
            shifts = -ops.sum(ops.cast(prompt_mask, "int"), axis=1) % seq_len
            ix = ops.arange(seq_len, dtype="int")
            ix = ops.expand_dims(ix, axis=0) - ops.expand_dims(shifts, axis=1)

            # This produces the desired shift (in fact a rollover).
            def roll_sequence(seq):
                return ops.take_along_axis(seq, ix, axis=1)

            # The shifting rolls the content over so the prompt is at the end of
            # the sequence and the generated text is at the beginning. We mask
            # it to retain the generated text only.
            y["padding_mask"] = ops.logical_xor(
                roll_sequence(prompt_mask), roll_sequence(x["padding_mask"])
            )
            # we assume the mask is enough and there is no need to zero-out the
            # values
            y["token_ids"] = roll_sequence(x["token_ids"])

            return y

        def postprocess(x):
            return self.preprocessor.generate_postprocess(x)

        # Normalize inputs, apply our three passes, and normalize outputs.
        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)

        if self.preprocessor is not None:
            inputs = [preprocess(x) for x in inputs]

        if strip_prompt:
            outputs = [strip_prompt_function(generate(x), x) for x in inputs]
        else:
            outputs = [generate(x) for x in inputs]

        if self.preprocessor is not None:
            outputs = [postprocess(x) for x in outputs]

        return self._normalize_generate_outputs(outputs, input_is_scalar)

    def export_to_transformers(self, path):
        """Export the full CausalLM model to HuggingFace Transformers format.

        This exports the trainable model, tokenizer, and configurations in a
        format compatible with HuggingFace Transformers. For unsupported model
        architectures, a ValueError is raised.

        If the preprocessor is attached (default), both the trainable model and
        tokenizer are exported. To export only the trainable model, set
        `self.preprocessor = None` before calling this method, then export the
        preprocessor separately via `preprocessor.export_to_transformers(path)`.

        Args:
            path: str. Path to save the exported model.
        """
        from keras_hub.src.utils.transformers.export.hf_exporter import (
            export_to_safetensors,
        )

        export_to_safetensors(self, path)

    def _post_quantize(self, mode, **kwargs):
        super()._post_quantize(mode, **kwargs)
        # Reset the compiled generate function.
        self.generate_function = None

    def get_quantization_layer_structure(self, mode):
        if mode not in ["gptq", "awq"]:
            return None

        backbone = self.backbone
        # Check for standard backbone structure.
        if not hasattr(backbone, "transformer_layers"):
            return None

        # Check for embedding.
        embedding = getattr(backbone, "token_embedding", None)
        if embedding is None:
            embedding = getattr(backbone, "embedding", None)

        if embedding is None:
            return None

        return {
            "pre_block_layers": [embedding],
            "sequential_blocks": backbone.transformer_layers,
        }
