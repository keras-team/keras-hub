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
from keras_nlp.src.models.bloom.bloom_backbone import BloomBackbone
from keras_nlp.src.models.bloom.bloom_causal_lm_preprocessor import (
    BloomCausalLMPreprocessor,
)
from keras_nlp.src.models.causal_lm import CausalLM


@keras_nlp_export("keras_nlp.models.BloomCausalLM")
class BloomCausalLM(CausalLM):
    """An end-to-end BLOOM model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a BLOOM model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Args:
        backbone: A `keras_nlp.models.BloomBackbone` instance.
        preprocessor: A `keras_nlp.models.BloomCausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    bloom_lm = keras_nlp.models.BloomCausalLM.from_preset("bloom_560m_multi")
    bloom_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    bloom_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    bloom_lm = keras_nlp.models.BloomCausalLM.from_preset("bloom_560m_multi")
    bloom_lm.compile(sampler="top_k")
    bloom_lm.generate("I want to say", max_length=30)

    bloom_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
    bloom_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {
        # Token ids for "<s> Keras is".
        "token_ids": np.array([[1, 46, 15762, 632, 3, 3, 3, 3, 3]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0]] * 2),
    }

    bloom_lm = keras_nlp.models.BloomCausalLM.from_preset(
        "bloom_560m_multi",
        preprocessor=None,
    )
    bloom_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    bloom_lm = keras_nlp.models.BloomCausalLM.from_preset("bloom_560m_multi")
    bloom_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        # Token ids for "<bos> Keras is deep learning library<eos>"
        "token_ids": np.array([[2, 214064, 603, 5271, 6044, 9581, 1, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
    }
    y = np.array([[214064, 603, 5271, 6044, 9581, 3, 0, 0]] * 2)
    sw = np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)

    bloom_lm = keras_nlp.models.BloomCausalLM.from_preset(
        "bloom_560m_multi",
        preprocessor=None,
    )
    bloom_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = [
        " airplane at airport",
        " airplane airport",
    ]
    vocab = ["<unk>", "<s>", "</s>", "<pad>"]
    vocab += ["!", "air", "Ġair", "plane", "Ġat", "port"]
    vocab = dict([(token, i) for i, token in enumerate(vocab)])
    merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
    merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
    merges += ["Ġai r", "Ġa i", "pla ne"]
    tokenizer = keras_nlp.models.BloomTokenizer(vocabulary=vocab, merges=merges)
    preprocessor = keras_nlp.models.BloomCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.BloomBackbone(
        vocabulary_size=tokenizer.vocabulary_size(),
        num_layers=4,
        num_heads=4,
        hidden_dim=32,
        intermediate_dim=128,
    )
    bloom_lm = keras_nlp.models.BloomCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    bloom_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = BloomBackbone
    preprocessor_cls = BloomCausalLMPreprocessor

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
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def build_cache(self, batch_size, max_length):
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        return ops.zeros(shape, dtype=self.compute_dtype)

    def call_with_cache(self, token_ids, cache, index):
        x = self.backbone.token_embedding(token_ids)
        x = self.backbone.embeddings_layer_norm(x)
        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                cache=current_cache,
                cache_update_index=index,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache
