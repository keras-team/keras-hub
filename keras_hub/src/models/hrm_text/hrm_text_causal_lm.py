"""Causal language-model task for HRM-Text."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.models.hrm_text.hrm_text_causal_lm_preprocessor import (
    HrmTextCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.HrmTextCausalLM")
class HrmTextCausalLM(CausalLM):
    """End-to-end HRM-Text model for causal language modeling.

    This task pairs :class:`HrmTextBackbone` with an LM head and KerasHub's
    sampler interface. It supports ordinary causal training as well as the
    PrefixLM format used to pretrain HRM-Text. When a preprocessor is attached,
    plain strings are causal examples and dictionaries with ``prefix`` and
    ``response`` are packed as PrefixLM examples.

    The official 1B weights are not bundled with the source distribution. Use
    the conversion script to create a local preset, then load it with
    :meth:`from_preset`.

    Args:
        backbone: An instance of `keras_hub.models.HrmTextBackbone`.
        preprocessor: An optional
            `keras_hub.models.HrmTextCausalLMPreprocessor`. If set, string
            inputs are preprocessed during `fit()`, `evaluate()`, `predict()`,
            and `generate()`.

    Examples:

    Generate from a converted local preset.
    ```python
    model = keras_hub.models.HrmTextCausalLM.from_preset(
        "/path/to/hrm_text_1b"
    )
    model.compile(sampler="greedy")
    model.generate("Summarize: Keras runs on multiple backends.", max_length=64)
    ```

    Train with PrefixLM inputs. Prefix tokens attend bidirectionally, while
    response tokens are trained causally.
    ```python
    examples = {
        "prefix": ["Question: What is 2 + 2?\\nAnswer:"],
        "response": [" 4"],
    }
    model.fit(examples, batch_size=1)
    ```
    """

    backbone_cls = HrmTextBackbone
    preprocessor_cls = HrmTextCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        self.backbone = backbone
        self.preprocessor = preprocessor
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def call_with_cache(
        self, token_ids, cache, cache_update_index, token_type_ids=None
    ):
        hidden_states, cache = self.backbone.call_with_cache(
            token_ids,
            cache,
            cache_update_index,
            token_type_ids=token_type_ids,
        )
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states, cache

    def make_generate_function(self):
        """Build a TensorFlow graph without AutoGraph source conversion.

        HRM-Text's recurrence uses a fixed Python loop and backend-native
        `ops.while_loop` in the sampler. Disabling AutoGraph avoids retracing
        the recurrent cache update closure while preserving graph execution.
        """
        if (
            keras.config.backend() == "tensorflow"
            and not self.run_eagerly
            and self.generate_function is None
        ):
            import tensorflow as tf

            self.generate_function = tf.function(
                self.generate_step,
                autograph=False,
                # Dynamic cache indices in `ops.while_loop` are not XLA
                # compilable on TensorFlow. Keep graph execution enabled.
                jit_compile=False,
            )
        return super().make_generate_function()

    def _build_cache(self, token_ids, token_type_ids):
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        backbone = self.backbone
        cache = ops.zeros(
            [
                batch_size,
                backbone.cache_slots,
                2,
                max_length,
                backbone.num_attention_heads,
                backbone.head_dim,
            ],
            dtype=self.compute_dtype,
        )
        _, hidden_states, cache = self.call_with_cache(
            token_ids, cache, 0, token_type_ids=token_type_ids
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = ops.cast(padding_mask, "int32")
        hidden_states, cache = self._build_cache(token_ids, token_type_ids)
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt, cache, cache_update_index
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
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
        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            padding_mask = ops.logical_not(
                ops.cast(cumsum - end_locations, "bool")
            )
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": ops.cast(padding_mask, token_ids.dtype),
            "token_type_ids": ops.cast(padding_mask, token_ids.dtype),
        }
