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
from keras_nlp.src.models.bart.bart_backbone import BartBackbone
from keras_nlp.src.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_nlp.src.models.seq_2_seq_lm import Seq2SeqLM


@keras_nlp_export("keras_nlp.models.BartSeq2SeqLM")
class BartSeq2SeqLM(Seq2SeqLM):
    """An end-to-end BART model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is used for
    conditional text generation. The encoder is given a "context" text (fed to
    the encoder), and the decoder predicts the next token based on both the
    encoder inputs and the previous tokens. You can finetune `BartSeq2SeqLM` to
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
        backbone: A `keras_nlp.models.BartBackbone` instance.
        preprocessor: A `keras_nlp.models.BartSeq2SeqLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation, given an input context.
    ```python
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.generate("The quick brown fox", max_length=30)

    # Generate with batched inputs.
    bart_lm.generate(["The quick brown fox", "The whale"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.compile(sampler="greedy")
    bart_lm.generate("The quick brown fox", max_length=30)
    ```

    Use `generate()` with encoder inputs and an incomplete decoder input (prompt).
    ```python
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.generate(
        {
            "encoder_text": "The quick brown fox",
            "decoder_text": "The fast"
        }
    )
    ```

    Use `generate()` without preprocessing.
    ```python
    # Preprocessed inputs, with encoder inputs corresponding to
    # "The quick brown fox", and the decoder inputs to "The fast". Use
    # `"padding_mask"` to indicate values that should not be overridden.
    prompt = {
        "encoder_token_ids": np.array([[0, 133, 2119, 6219, 23602, 2, 1, 1]]),
        "encoder_padding_mask": np.array(
            [[True, True, True, True, True, True, False, False]]
        ),
        "decoder_token_ids": np.array([[2, 0, 133, 1769, 2, 1, 1]]),
        "decoder_padding_mask": np.array([[True, True, True, True, False, False]])
    }

    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
        "bart_base_en",
        preprocessor=None,
    )
    bart_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = {
        "encoder_text": ["The quick brown fox jumped.", "I forgot my homework."],
        "decoder_text": ["The fast hazel fox leapt.", "I forgot my assignment."]
    }
    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "encoder_token_ids": np.array([[0, 133, 2119, 2, 1]] * 2),
        "encoder_padding_mask": np.array([[1, 1, 1, 1, 0]] * 2),
        "decoder_token_ids": np.array([[2, 0, 133, 1769, 2]] * 2),
        "decoder_padding_mask": np.array([[1, 1, 1, 1, 1]] * 2),
    }
    y = np.array([[0, 133, 1769, 2, 1]] * 2)
    sw = np.array([[1, 1, 1, 1, 0]] * 2)

    bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
        "bart_base_en",
        preprocessor=None,
    )
    bart_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = {
        "encoder_text": [" afternoon sun"],
        "decoder_text": ["noon sun"],
    }
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "Ġafter": 5,
        "noon": 6,
        "Ġsun": 7,
    }
    merges = ["Ġ a", "Ġ s", "Ġ n", "e r", "n o", "o n", "Ġs u", "Ġa f", "no on"]
    merges += ["Ġsu n", "Ġaf t", "Ġaft er"]

    tokenizer = keras_nlp.models.BartTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.BartSeq2SeqLMPreprocessor(
        tokenizer=tokenizer,
        encoder_sequence_length=128,
        decoder_sequence_length=128,
    )
    backbone = keras_nlp.models.BartBackbone(
        vocabulary_size=50265,
        num_layers=6,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=128,
    )
    bart_lm = keras_nlp.models.BartSeq2SeqLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    bart_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = BartBackbone
    preprocessor_cls = BartSeq2SeqLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)["decoder_sequence_output"]
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def build_cache(self, batch_size, max_length):
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        return ops.zeros(shape, dtype=self.compute_dtype)

    def compute_cross_attention_cache(
        self, encoder_token_ids, encoder_padding_mask
    ):
        """Does a forward pass on the encoder and returns the encoder output."""
        # Embedding layers.
        tokens = self.backbone.token_embedding(encoder_token_ids)
        positions = self.backbone.encoder_position_embedding(tokens)
        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.encoder_embeddings_add((tokens, positions))
        x = self.backbone.encoder_embeddings_layer_norm(x)
        x = self.backbone.encoder_embeddings_dropout(x)
        # Transformer encoder layers.
        for layer in self.backbone.encoder_transformer_layers:
            x = layer(x, padding_mask=encoder_padding_mask)
        # Transformer encoder layers.
        caches = []
        for layer in self.backbone.decoder_transformer_layers:
            caches.append(layer.compute_cross_attention_cache(x))
        return ops.stack(caches, axis=1)

    def call_with_cache(
        self,
        token_ids,
        cache,
        index,
        *,
        encoder_padding_mask,
        cross_attention_cache,
    ):
        tokens = self.backbone.token_embedding(token_ids)
        positions = self.backbone.decoder_position_embedding(
            tokens, start_index=index
        )
        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.decoder_embeddings_add((tokens, positions))
        x = self.backbone.decoder_embeddings_layer_norm(x)
        x = self.backbone.decoder_embeddings_dropout(x)
        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, layer in enumerate(self.backbone.decoder_transformer_layers):
            current_self_attention_cache = cache[:, i, ...]
            current_cross_attention_cache = cross_attention_cache[:, i, ...]
            x, next_cache, _ = layer(
                decoder_sequence=x,
                encoder_padding_mask=encoder_padding_mask,
                self_attention_cache=current_self_attention_cache,
                self_attention_cache_update_index=index,
                cross_attention_cache=current_cross_attention_cache,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (
            logits,
            hidden_states,
            cache,
        )
