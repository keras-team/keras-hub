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
"""Whisper Seq2Seq LM (Language Model)."""

import copy
import os

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.generative_task import GenerativeTask
from keras_nlp.models.whisper.whisper_backbone import Padder
from keras_nlp.models.whisper.whisper_backbone import WhisperBackbone
from keras_nlp.models.whisper.whisper_presets import backbone_presets
from keras_nlp.models.whisper.whisper_seq_2_seq_lm_preprocessor import (
    WhisperSeq2SeqLMPreprocessor,
)
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras_nlp_export("keras_nlp.models.WhisperSeq2SeqLM")
class WhisperSeq2SeqLM(GenerativeTask):
    """An end-to-end Whisper model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is used for
    conditional text generation. The encoder is given a "context" text (fed to
    the encoder), and the decoder predicts the next token based on both the
    encoder inputs and the previous tokens. You can finetune `WhisperSeq2SeqLM` to
    generate text for any seq2seq task (e.g., translation or summarization).

    This model has a `generate()` method, which generates text based on
    encoder inputs and an optional prompt for the decoder. The generation
    strategy used is controlled by an additional `sampler` argument passed to
    `compile()`. You can recompile the model with different `keras_nlp.samplers`
    objects to control the generation. By default, `"top_k"` sampling will be
    used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        backbone: A `keras_nlp.models.WhisperBackbone` instance.
        preprocessor: A `keras_nlp.models.WhisperSeq2SeqLMPreprocessor` or `None`.
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
        hidden_states = backbone(inputs)["decoder_sequence_output"]
        outputs = backbone.token_embedding(hidden_states, reverse=True)

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
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            jit_compile=True,
        )

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return WhisperBackbone

    @classproperty
    def preprocessor_cls(cls):
        return WhisperSeq2SeqLMPreprocessor

    def call_decoder_with_cache(
        self,
        encoder_hidden_states,
        decoder_token_ids,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
    ):
        """Forward pass with a key/value caches for generative decoding..

        `call_decoder_with_cache` adds an additional inference-time forward pass
        for the model for seq2seq text generation. Unlike calling the model
        directly, this method does two things to optimize text generation:

        - Allows caching previous key/value tensors in the decoder's
          self-attention layer to avoid recomputing the outputs of seen tokens.
        - Allows caching key/value tensors in the decoder's cross-attention
          layer to avoid recomputing the encoder outputs.

        Args:
            encoder_hidden_states: a dense float Tensor of shape
                `(batch_size, encoder_sequence_length, hidden_dim)`. The
                sequence of hidden states at the output of the encoder's last
                layer.
            decoder_token_ids: a dense int Tensor of shape
                `(batch_size, max_length)`. Input token ids to be fed to
                the decoder.
            self_attention_cache: a dense float Tensor of shape
                `(batch_size, num_layers, 2, max_length, num_heads, key_dims)`.
                The cached key/value tensors of previously seen tokens in the
                decoder's self-attention layer.
            self_attention_cache_update_index: an int or int Tensor, the index
                at which to update the `self_attention_cache`. Usually, this is
                the index of the current token being processed during decoding.
            cross_attention_cache: a dense float Tensor of shape
                `(batch_size, num_layers, 2, encoder_sequence_length, num_heads, key_dims)`.
                The cached key/value tensors of the encoder outputs in the
                decoder's cross-attention layer.
            cross_attention_cache_update_index: an int or int Tensor, the index
                at which to update the `cross_attention_cache`. Usually, this is
                either `0` (compute the entire `cross_attention_cache`), or
                `None` (reuse a previously computed `cross_attention_cache`).

        Returns:
            A `(logits, hidden_states, self_attention_cache, cross_attention_cache)`
            tuple, where `logits` is the language model logits for the input
            `decoder_token_ids`, `hidden_states` is the final hidden
            representation of the input tokens, `self_attention_cache` is the
            key/value cache in the decoder's self-attention layer and
            `cross_attention_cache` is the key/value cache in the decoder's
            cross-attention layer.
        """
        # Embedding layers.
        x = self.backbone.get_layer("decoder_token_and_position_embedding")(
            decoder_token_ids
        )

        # Apply dropout to embeddings.
        x = self.backbone.get_layer("decoder_embeddings_dropout")(x)

        # Every decoder layer has a separate cache for the self-attention layer
        # and the cross-attention layer. We update all of them separately.
        self_attention_caches = []
        cross_attention_caches = []
        for i in range(self.backbone.num_layers):
            current_self_attention_cache = self_attention_cache[:, i, ...]
            current_cross_attention_cache = cross_attention_cache[:, i, ...]

            (
                x,
                next_self_attention_cache,
                next_cross_attention_cache,
            ) = self.backbone.get_layer(f"transformer_decoder_layer_{i}")(
                decoder_sequence=x,
                encoder_sequence=encoder_hidden_states,
                self_attention_cache=current_self_attention_cache,
                self_attention_cache_update_index=self_attention_cache_update_index,
                cross_attention_cache=current_cross_attention_cache,
                cross_attention_cache_update_index=cross_attention_cache_update_index,
            )

            if self_attention_cache_update_index is not None:
                self_attention_caches.append(next_self_attention_cache)
            if cross_attention_cache_update_index is not None:
                cross_attention_caches.append(next_cross_attention_cache)

        if self_attention_cache_update_index is not None:
            self_attention_cache = ops.stack(self_attention_caches, axis=1)
        if cross_attention_cache_update_index is not None:
            cross_attention_cache = ops.stack(cross_attention_caches, axis=1)

        x = self.backbone.get_layer("decoder_layer_norm")(x)

        hidden_states = x
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def call_encoder(self, features):
        """Does a forward pass on the encoder and returns the encoder output."""

        # Embedding layers.
        embedded_features = self.backbone.get_layer(
            "encoder_token_embedding_conv_layer_1"
        )(features)
        embedded_features = keras.activations.gelu(
            embedded_features, approximate=False
        )
        embedded_features = Padder()(embedded_features)
        embedded_features = self.backbone.get_layer(
            "encoder_token_embedding_conv_layer_2"
        )(embedded_features)
        embedded_features = keras.activations.gelu(
            embedded_features, approximate=False
        )
        position_embedding = self.backbone.get_layer(
            "encoder_position_embedding"
        )(embedded_features)

        # Sum, normalize and apply dropout to embeddings.
        x = keras.layers.Add()((embedded_features, position_embedding))
        x = self.backbone.get_layer("encoder_embeddings_dropout")(x)

        # Transformer encoder layers.
        for i in range(self.backbone.num_layers):
            x = self.backbone.get_layer(f"transformer_encoder_layer_{i}")(x)

        x = self.backbone.get_layer("encoder_layer_norm")(x)

        return x

    def _initialize_cache(self, encoder_features, decoder_token_ids):
        """Initializes empty self-attention cache and cross-attention cache."""
        batch_size = ops.shape(encoder_features)[0]
        encoder_max_length = ops.shape(encoder_features)[1]
        decoder_max_length = ops.shape(decoder_token_ids)[1]

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
        self_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)

        shape[3] = encoder_max_length
        cross_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)

        return (self_attention_cache, cross_attention_cache)

    def _build_cache(self, encoder_features, decoder_token_ids):
        """Builds the self-attention cache and the cross-attention cache (key/value pairs)."""
        encoder_hidden_states = self.call_encoder(features=encoder_features)
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            encoder_features, decoder_token_ids
        )

        # Seed the self-attention cache and the cross-attention cache.
        (
            _,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            decoder_token_ids=decoder_token_ids,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=0,
            cross_attention_cache=cross_attention_cache,
            cross_attention_cache_update_index=0,
        )
        return (
            hidden_states,
            encoder_hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def generate_step(
        self,
        inputs,
        end_token_id=None,
    ):
        """A compilable generation function for a batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"encoder_features"`,
        `"decoder_token_ids"` and `"decoder_padding_mask"`.

        Args:
            inputs: A dictionary with three keys - `"encoder_features"`,
                `"decoder_token_ids"` and `"decoder_padding_mask"`, with batched
                tensor values.
            end_token_id: The id of the end token to stop on. If all
                sequences have produced a new `end_token_id`, generation
                will stop.
        """
        (
            encoder_features,
            decoder_token_ids,
            decoder_padding_mask,
        ) = (
            inputs["encoder_features"],
            inputs["decoder_token_ids"],
            inputs["decoder_padding_mask"],
        )

        batch_size = ops.shape(encoder_features)[0]

        # Create and seed cache with a single forward pass.
        (
            hidden_states,
            encoder_hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self._build_cache(encoder_features, decoder_token_ids)
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_index = index - 1
            num_samples = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_index], [num_samples, 1])

            def repeat_tensor(x):
                """Repeats tensors along batch axis to match dim for beam search."""
                if ops.shape(x)[0] == num_samples:
                    return x
                return ops.repeat(x, repeats=num_samples // batch_size, axis=0)

            logits, hidden_states, cache, _ = self.call_decoder_with_cache(
                encoder_hidden_states=repeat_tensor(encoder_hidden_states),
                decoder_token_ids=prompt,
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                cross_attention_cache=repeat_tensor(cross_attention_cache),
                cross_attention_cache_update_index=None,
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        decoder_token_ids = self._sampler(
            next=next,
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
            # prompt (not in locations where `decoder_padding_mask` is True).
            end_locations = ops.logical_and(
                ops.equal(decoder_token_ids, end_token_id),
                ops.logical_not(decoder_padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after `end_locations`.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            decoder_padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            decoder_padding_mask = ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        language=None,
        task=None,
        no_timestamps=True,
        **kwargs,
    ):
        """Instantiate `WhisperSeq2SeqLM` model from preset architecture and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.
            language: string, language token (eg., `"<|en|>"`). Should only be
                passed if your tokenizer is multilingual.
            task: string, task name. One of `"transcribe"`, `"translate"`.
                Should only be passed if your tokenizer is multilingual.
            no_timestamps: bool. If True, `"<|no_timestamps|>"` will be added as
                a special token to your input.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = WhisperSeq2SeqLM.from_preset("{{example_preset_name}}")

        # Load randomly initialized model from preset architecture
        model = WhisperSeq2SeqLM.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
        ```
        """
        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )

        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        if "preprocessor" not in kwargs:
            kwargs["preprocessor"] = cls.preprocessor_cls.from_preset(
                preset,
                language=language,
                task=task,
                no_timestamps=no_timestamps,
            )

        # Check if preset is backbone-only model
        if preset in cls.backbone_cls.presets:
            backbone = cls.backbone_cls.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets
        metadata = cls.presets[preset]
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )

        model.load_weights(weights)
        return model


format_docstring(
    example_preset_name=next(iter(backbone_presets), ""),
    preset_names='", "'.join(backbone_presets),
)(WhisperSeq2SeqLM.from_preset.__func__)
