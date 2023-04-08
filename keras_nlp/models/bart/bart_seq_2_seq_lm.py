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
from keras_nlp.utils.tf_utils import truncate_at_token


@keras_nlp_export("keras_nlp.models.BartSeq2SeqLM")
class BartSeq2SeqLM(Task):
    """An end-to-end BART model for causal langauge modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens the next token based on previous tokens, which is the way BART gets
    pretrained. You can finetune `BartSeq2SeqLM` to generate text similar to
    the custom dataset.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/gpt-2).

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
        # Private sampler set by compile.
        self._sampler = get_sampler("top_k")

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
        encoder_cache,
        decoder_token_ids,
        decoder_cache,
        decoder_cache_index,
    ):
        """Forward pass of `BartSeq2SeqLM` with `encoder_cache` and `decoder_cache`.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            decoder_token_ids: a dense int Tensor, input token ids.
            decoder_cache: a dense float Tensor, the decoder_cache of key and value.
            decoder_cache_index: int, or int Tensor. The index of current inputs in the
                whole sequence.

        Returns:
            A (logits, decoder_cache) tuple. Where the first output is the language
            model logits for the input decoder_token_ids and the second output is the
            decoder_cache.
        """
        # Embedding layers.
        token_embedding = self.backbone.get_layer("token_embedding")(
            decoder_token_ids
        )
        position_embedding = self.backbone.get_layer(
            "decoder_position_embedding"
        )(token_embedding, start_index=decoder_cache_index)

        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.get_layer("decoder_embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = self.backbone.get_layer("decoder_embeddings_layer_norm")(x)
        # x = self.backbone.get_layer("decoder_embeddings_dropout")(x)

        # Each decoder layer has a decoder_cache; we update them separately.
        decoder_caches = tf.unstack(decoder_cache, axis=1)
        for i in range(self.backbone.num_layers):
            current_decoder_cache = decoder_caches[i]
            x, next_decoder_cache = self.backbone.get_layer(
                f"transformer_decoder_layer_{i}"
            )(
                x,
                encoder_sequence=encoder_cache,
                cache=current_decoder_cache,
                cache_index=decoder_cache_index,
            )
            decoder_caches[i] = next_decoder_cache
        decoder_cache = tf.stack(decoder_caches, axis=1)

        x = tf.matmul(
            x,
            self.backbone.get_layer("token_embedding").embeddings,
            transpose_b=True,
        )
        return x, decoder_cache

    def _build_encoder_cache(self, token_ids, padding_mask):
        """Builds a cache for the encoder outputs to avoid doing a forward pass
        on the encoder at every time step.
        """

        # Embedding layers.
        token_embedding = self.backbone.get_layer("token_embedding")(token_ids)
        position_embedding = self.backbone.get_layer("position_embedding")(
            token_embedding
        )

        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.get_layer("encoder_embeddings_add")(
            token_embedding, position_embedding
        )
        x = self.backbone.get_layer("encoder_embeddings_layer_norm")(x)
        # x = self.backbone.get_layer("encoder_embeddings_dropout")(x)

        # Transformer encoder layers.
        for i in range(self.backbone.num_layers):
            x = self.backbone.get_layer(f"transformer_encoder_layer_{i}")(
                x, padding_mask=padding_mask
            )

        cache = x
        return cache

    def _build_decoder_cache(self, prompt):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = tf.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        _, cache = self.call_with_cache(prompt, cache, 0)
        return cache

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

        def generate_function(
            encoder_token_ids,
            encoder_padding_mask,
            prompt,
            input_mask,
            min_length,
        ):
            # Create and seed cache with a single forward pass.
            encoder_cache = self._build_encoder_cache(
                encoder_token_ids, encoder_padding_mask
            )
            decoder_cache = self._build_decoder_cache(prompt)

            def next(prompt, state, index):
                # The cache index is the index of our previous token.
                cache_index = index - 1
                prompt = tf.slice(prompt, [0, cache_index], [-1, 1])
                logits, state = self.call_with_cache(
                    encoder_cache, prompt, state, cache_index
                )
                return tf.squeeze(logits, axis=1), state

            return self._sampler(
                next=next,
                prompt=prompt,
                state=decoder_cache,
                index=min_length,
                mask=input_mask,
                end_token_id=self.preprocessor.tokenizer.end_token_id,
            )

        if self.run_eagerly:
            self.generate_function = generate_function
        else:
            # `jit_compile` is a property of keras.Model after TF 2.12.
            # Use `getattr()` for backwards compatibility.
            jit_compile = getattr(self, "jit_compile", True)
            self.generate_function = tf.function(
                generate_function, jit_compile=jit_compile
            )
        return self.generate_function

    def generate(
        self,
        encoder_text,
        prompt,
        max_length,
    ):
        """Generate text.

        This method generates text based on given `encoder_text` and `prompt`.
        Generation will continue until `max_length` is met, and all tokens
        generated after `end_token` will be truncated. The sampling strategy can
        be set in the `compile` method.

        Args:
            encoder_text: a string, string Tensor or string RaggedTensor. The
                input to the encoder, i.e., the context. The generated text is
                conditioned on this input.
            prompt: a string, string Tensor or string RaggedTensor. The prompt
                text for generation.
            max_length: int. The max length of generated sequence.
        """
        if self.preprocessor is None:
            raise ValueError(
                "`self.preprocessor` is `None`, please make sure "
                "`preprocessor` is set before calling `generate`."
            )

        # Tokenize the encoder inputs. We can use the preprocessor directly
        # here.
        encoder_text = tf.convert_to_tensor(encoder_text)
        encoder_text_is_scalar = encoder_text.shape.rank == 0
        encoder_text = (
            encoder_text[tf.newaxis] if encoder_text_is_scalar else encoder_text
        )
        preprocessed_inputs = self.preprocessor(
            {
                "encoder_text": encoder_text,
                "decoder_text": "dummy text",
            }
        )
        encoder_token_ids = preprocessed_inputs[0]["encoder_token_ids"]
        encoder_padding_mask = preprocessed_inputs[0]["encoder_padding_mask"]

        # Tokenize the prompt. We cannot use the preprocessor directly since
        # the `max_length` might be different. Moreover, the
        # `BartSeq2SeqLMPreprocessor` handles the training case.
        prompt = tf.convert_to_tensor(prompt)
        input_is_scalar = prompt.shape.rank == 0
        prompt = prompt[tf.newaxis] if input_is_scalar else prompt

        prompt = self.preprocessor.tokenizer(prompt)

        # Pad ragged to dense tensors.
        padded_shape = (None, max_length)
        min_length = tf.reduce_min(prompt.row_lengths())
        input_mask = tf.ones_like(prompt, tf.bool).to_tensor(shape=padded_shape)
        prompt = prompt.to_tensor(shape=padded_shape)

        # Run the (possibly compiled) generate function on dense inputs.
        generate_function = self.make_generate_function()
        output = generate_function(
            encoder_token_ids,
            encoder_padding_mask,
            prompt,
            input_mask,
            min_length,
        )

        # Truncate to ragged by removing tokens after the first end token.
        end_token_id = self.preprocessor.tokenizer.end_token_id
        output = truncate_at_token(output, end_token_id, input_mask)

        # Detokenize.
        output = self.preprocessor.tokenizer.detokenize(output)
        return tf.squeeze(output, 0) if input_is_scalar else output
