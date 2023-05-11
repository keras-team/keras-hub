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
"""BART Seq2Seq LM (Language Model)."""

import copy

import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.bart.bart_backbone import BartBackbone
from keras_nlp.models.bart.bart_presets import backbone_presets
from keras_nlp.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_nlp.models.task import Task
from keras_nlp.samplers.serialization import get as get_sampler
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.tf_utils import tensor_to_string_list


@keras_nlp_export("keras_nlp.models.BartSeq2SeqLM")
class BartSeq2SeqLM(Task):
    """An end-to-end BART model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is used for
    conditional text generation. The encoder is given a "context" text, and the
    decoder predicts the next token based on both the encoder inputs and the
    previous tokens. You can finetune `BartSeq2SeqLM` to generate text similar
    to the custom dataset.

    This model has a `generate()` method, which generates text based on a
    encoder inputs and a prompt. The generation strategy used is controlled by
    an additional `sampler` argument on `compile()`. You can recompile the model
    with different `keras_nlp.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        backbone: A `keras_nlp.models.BartBackbone` instance.
        preprocessor: A `keras_nlp.models.BartSeq2SeqLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.
    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        x = backbone(inputs)["decoder_sequence_output"]
        # Use token embedding weights to project from the token representation
        # to vocabulary logits.
        outputs = tf.matmul(
            x,
            backbone.token_embedding.embeddings,
            transpose_b=True,
        )

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )

        self.backbone = backbone
        self.preprocessor = preprocessor
        self.generate_function = None
        self._sampler = None

        # Default compilation
        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(2e-5),
            metrics=keras.metrics.SparseCategoricalAccuracy(),
            jit_compile=is_xla_compatible(self),
        )

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return BartBackbone

    @classproperty
    def preprocessor_cls(cls):
        return BartSeq2SeqLMPreprocessor

    def call_with_cache(
        self,
        encoder_outputs,
        encoder_padding_mask,
        decoder_token_ids,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
    ):
        """Forward pass of `BartSeq2SeqLM` with `self_attention_cache` and `cross_attention_cache`.

        `call_with_cache` adds an additional inference-time forward pass for the
        model for seq2seq text generation. Unlike calling the model directly,
        this method does two things to optimize text generation:

        - Allows caching previous key/value tensors in the decoder's
          self-attention layer to avoid recomputing the outputs of seen tokens.
        - Allows caching the key/value tensors in the decoder's cross-attention
          layer to avoid recomputing the encoder outputs.

        Args:
            encoder_outputs: a dense float Tensor. The encoder output.
            encoder_padding_mask: a dense float Tensor. The padding mask for
                the encoder input.
            decoder_token_ids: a dense int Tensor, input token ids to be fed to
                the decoder.
            self_attention_cache: a dense float Tensor, the cached key/value
                tensors of previously seen tokens in the decoder's self-attention
                layer.
            self_attention_cache_update_index: int, or int Tensor. The index of
                current inputs in the whole decoder sequence.
            cross_attention_cache: a dense float Tensor, the cached key/value
                tensors of the encoder outputs in the decoder's cross-attention
                layer.
            cross_attention_cache_update_index: int, or int Tensor. The index of
                current inputs in the whole encoder sequence. This either takes
                value 0, or None. The former means that the entire
                cross-attention cache is updated in one go (since we don't need
                to update it token-by-token), while the latter means that
                `cross_attention_cache` will be passed through without any
                changes.

        Returns:
            A `(logits, hidden_states, self_attention_cache, cross_attention_cache,)`
            tuple, where `logits` is the language model logits for the input
            `decoder_token_ids`, `hidden_states` is the final hidden
            representation of the input tokens, `self_attention_cache` is the
            key/value cache in the decoder's self-attention layer and
            `cross_attention_cache` is the key/value cache in the decoder's
            cross-attention layer.
        """
        # Embedding layers.
        token_embedding = self.backbone.get_layer("token_embedding")(
            decoder_token_ids
        )
        position_embedding = self.backbone.get_layer(
            "decoder_position_embedding"
        )(token_embedding, start_index=self_attention_cache_update_index)

        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.get_layer("decoder_embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = self.backbone.get_layer("decoder_embeddings_layer_norm")(x)
        x = self.backbone.get_layer("decoder_embeddings_dropout")(x)

        # Every decoder layer has a separate cache for the self-attention layer
        # and the cross-attention layer. We update all of them separately.
        self_attention_caches = tf.unstack(self_attention_cache, axis=1)
        cross_attention_caches = tf.unstack(cross_attention_cache, axis=1)
        for i in range(self.backbone.num_layers):
            current_self_attention_cache = self_attention_caches[i]
            current_cross_attention_cache = cross_attention_caches[i]

            (
                x,
                next_self_attention_cache,
                next_cross_attention_cache,
            ) = self.backbone.get_layer(f"transformer_decoder_layer_{i}")(
                decoder_sequence=x,
                encoder_sequence=encoder_outputs,
                encoder_padding_mask=encoder_padding_mask,
                self_attention_cache=current_self_attention_cache,
                self_attention_cache_update_index=self_attention_cache_update_index,
                cross_attention_cache=current_cross_attention_cache,
                cross_attention_cache_update_index=cross_attention_cache_update_index,
            )

            self_attention_caches[i] = next_self_attention_cache
            if cross_attention_cache_update_index is not None:
                cross_attention_caches[i] = next_cross_attention_cache

        self_attention_cache = tf.stack(self_attention_caches, axis=1)
        if cross_attention_cache_update_index is not None:
            cross_attention_cache = tf.stack(cross_attention_caches, axis=1)

        hidden_states = x

        logits = tf.matmul(
            hidden_states,
            self.backbone.get_layer("token_embedding").embeddings,
            transpose_b=True,
        )
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def _get_encoder_outputs(self, token_ids, padding_mask):
        """Does a forward pass on the encoder and returns the encoder output."""

        # Embedding layers.
        token_embedding = self.backbone.get_layer("token_embedding")(token_ids)
        position_embedding = self.backbone.get_layer(
            "encoder_position_embedding"
        )(token_embedding)

        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.get_layer("encoder_embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = self.backbone.get_layer("encoder_embeddings_layer_norm")(x)
        x = self.backbone.get_layer("encoder_embeddings_dropout")(x)

        # Transformer encoder layers.
        for i in range(self.backbone.num_layers):
            x = self.backbone.get_layer(f"transformer_encoder_layer_{i}")(
                x, padding_mask=padding_mask
            )

        return x

    def _initialize_cache(self, encoder_token_ids, decoder_token_ids):
        """Initializes empty self-attention cache and cross-attention cache."""
        batch_size = tf.shape(encoder_token_ids)[0]
        encoder_max_length = tf.shape(encoder_token_ids)[1]
        decoder_max_length = tf.shape(decoder_token_ids)[1]

        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads

        shape = [
            batch_size,
            num_layers,
            2,
            decoder_max_length,
            num_heads,
            head_dim,
        ]
        self_attention_cache = tf.zeros(shape, dtype=self.compute_dtype)

        shape[3] = encoder_max_length
        cross_attention_cache = tf.zeros(shape, dtype=self.compute_dtype)

        return (self_attention_cache, cross_attention_cache)

    def _build_cache(
        self, encoder_token_ids, encoder_padding_mask, decoder_token_ids
    ):
        """Builds the self-attention cache and the cross-attention cache (key/value pairs)."""
        encoder_outputs = self._get_encoder_outputs(
            token_ids=encoder_token_ids, padding_mask=encoder_padding_mask
        )
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            encoder_token_ids, decoder_token_ids
        )

        # Seed the self-attention cache and the cross-attention cache.
        (
            _,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self.call_with_cache(
            encoder_outputs=encoder_outputs,
            encoder_padding_mask=encoder_padding_mask,
            decoder_token_ids=decoder_token_ids,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=0,
            cross_attention_cache=cross_attention_cache,
            cross_attention_cache_update_index=0,
        )
        return (
            hidden_states,
            encoder_outputs,
            self_attention_cache,
            cross_attention_cache,
        )

    def compile(
        self,
        *args,
        run_eagerly=False,
        jit_compile=True,
        sampler="top_k",
        **kwargs,
    ):
        xla_compatible = is_xla_compatible(self)
        super().compile(
            *args,
            run_eagerly=run_eagerly,
            # Only `jit_compile` if not eager and in a compatible environment.
            jit_compile=jit_compile and xla_compatible and not run_eagerly,
            **kwargs,
        )
        self._sampler = get_sampler(sampler)
        # Clear the compiled generate function.
        self.generate_function = None

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        if self.run_eagerly:
            self.generate_function = self.generate_step
        else:
            # `jit_compile` is a property of keras.Model after TF 2.12.
            # Use `getattr()` for backwards compatibility.
            jit_compile = getattr(self, "jit_compile", True)
            self.generate_function = tf.function(
                self.generate_step, jit_compile=jit_compile
            )
        return self.generate_function

    def generate_step(
        self,
        inputs,
        end_token_id=None,
    ):
        """A compilable generation function.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"encoder_token_ids"`,
        `"encoder_padding_mask"`, `"decoder_token_ids"` and
        `"decoder_padding_mask"`.

        Args:
            inputs: A dictionary with four keys - `"encoder_token_ids"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"` and
                `"decoder_padding_mask"`, with batched tensor values.
            end_token_id: The id of the end token to stop on. If all
                sequences have produced a new `end_token_id`, generation
                will stop.
        """
        (
            encoder_token_ids,
            encoder_padding_mask,
            decoder_token_ids,
            decoder_padding_mask,
        ) = (
            inputs["encoder_token_ids"],
            inputs["encoder_padding_mask"],
            inputs["decoder_token_ids"],
            inputs["decoder_padding_mask"],
        )

        # Create and seed cache with a single forward pass.
        (
            hidden_states,
            encoder_outputs,
            self_attention_cache,
            cross_attention_cache,
        ) = self._build_cache(
            encoder_token_ids, encoder_padding_mask, decoder_token_ids
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = tf.math.reduce_sum(
            tf.cast(decoder_padding_mask, tf.int32), axis=-1
        )
        # Start at the first index that has no user inputted id.
        index = tf.math.reduce_min(row_lengths)

        def next_token(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_index = index - 1
            prompt = tf.slice(prompt, [0, cache_index], [-1, 1])

            logits, hidden_states, cache, _ = self.call_with_cache(
                encoder_outputs=encoder_outputs,
                encoder_padding_mask=encoder_padding_mask,
                decoder_token_ids=prompt,
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                cross_attention_cache=cross_attention_cache,
                cross_attention_cache_update_index=None,
            )
            return (
                tf.squeeze(logits, axis=1),
                tf.squeeze(hidden_states, axis=1),
                cache,
            )

        decoder_token_ids = self._sampler(
            next=next_token,
            prompt=decoder_token_ids,
            cache=self_attention_cache,
            index=index,
            mask=decoder_padding_mask,
            end_token_id=end_token_id,
            hidden_states=hidden_states,
        )

        # Compute an output padding mask with the token ids we updated.
        if end_token_id is not None:
            # Build a mask of `end_token_id` locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = (decoder_token_ids == end_token_id) & (
                ~decoder_padding_mask
            )
            end_locations = tf.cast(end_locations, tf.int32)
            # Use cumsum to get ones in all locations after end_locations.
            overflow = tf.math.cumsum(end_locations, exclusive=True)
            # Our padding mask is the inverse of these overflow locations.
            decoder_padding_mask = ~tf.cast(overflow, tf.bool)
        else:
            # Without early stopping, all locations will have been updated.
            decoder_padding_mask = tf.ones_like(
                decoder_token_ids, dtype=tf.bool
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    def _normalize_generate_inputs(
        self,
        inputs,
    ):
        """Normalize user input to the generate function.

        This function coverts all inputs to tensors, adds a batch dimension if
        necessary, and returns a iterable "dataset like" object (either an
        actual `tf.data.Dataset` or a list with a single batch element).
        """
        input_is_scalar = False

        if isinstance(inputs, tf.data.Dataset):
            return inputs, input_is_scalar

        def normalize(x):
            x_is_scalar = False
            if isinstance(x, str) or isinstance(x, list):
                x = tf.convert_to_tensor(x)

            if isinstance(x, tf.Tensor) and x.shape.rank == 0:
                x_is_scalar = True
                x = x[tf.newaxis]

            return x, x_is_scalar

        for key in inputs:
            inputs[key], input_is_scalar = normalize(inputs[key])

        # We avoid converting to a dataset purely for speed, for a single batch
        # of input, creating a dataset would add significant overhead.
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
            x = tf.concat(x, axis=0)
            x = tf.squeeze(x, 0) if input_is_scalar else x
            is_string = x.dtype == tf.string
            # Convert outputs to a friendly pythonic type. For numerical outputs
            # that is numpy, for string outputs that is `list` and `str`.
            return tensor_to_string_list(x) if is_string else x.numpy()

        if isinstance(outputs[0], dict):
            return {
                "decoder_token_ids": normalize(
                    [x["decoder_token_ids"] for x in outputs]
                ),
                "decoder_padding_mask": normalize(
                    [x["decoder_padding_mask"] for x in outputs]
                ),
            }
        return normalize([x for x in outputs])

    def generate(
        self,
        inputs,
        max_length=None,
    ):
        """Generates text conditioned on the encoder inputs.

        This method generates text based on given `encoder_text` and `prompt`.
        Generation will continue until `max_length` is met, and all tokens
        generated after `end_token` will be truncated. The sampling strategy can
        be set in the `compile` method.

        Args:
            encoder_text: a string, string Tensor or string RaggedTensor. The
                input to the encoder, i.e., the context. The generated text is
                conditioned on this input.
            prompt: a string, string Tensor or string RaggedTensor. The prompt
                text for generation. This is fed as input to the decoder.
            max_length: int. The max length of generated sequence.
            add_start_token: bool. Whether to add the start token to `prompt`.
        """
        # Setup our three main passes.
        # 1. Optionally preprocessing strings to dense integer tensors.
        # 2. Generate new tokens via a compiled function on dense tensors.
        # 3. Optionally postprocess dense integer tensors back to string.
        generate_function = self.make_generate_function()
        end_token_id = None
        if self.preprocessor is not None:
            end_token_id = self.preprocessor.tokenizer.end_token_id

        def preprocess(x):
            return self.preprocessor.generate_preprocess(
                x, sequence_length=max_length
            )

        def generate(x):
            return generate_function(x, end_token_id=end_token_id)

        def postprocess(x):
            return self.preprocessor.generate_postprocess(x)

        # Normalize inputs, apply our three passes, and normalize outputs.
        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)

        if self.preprocessor is not None:
            if isinstance(inputs, tf.data.Dataset):
                inputs = inputs.map(preprocess, tf.data.AUTOTUNE)
                inputs = inputs.prefetch(tf.data.AUTOTUNE)
            else:
                # Fast path for non-dataset, single-batch input.
                inputs = [preprocess(x) for x in inputs]

        outputs = [generate(x) for x in inputs]

        if self.preprocessor is not None:
            outputs = [postprocess(x) for x in outputs]

        return self._normalize_generate_outputs(outputs, input_is_scalar)
