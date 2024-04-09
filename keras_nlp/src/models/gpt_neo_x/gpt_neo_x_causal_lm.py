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

from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.causal_lm import CausalLM
from keras_nlp.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_nlp.src.models.gpt_neo_x.gpt_neo_x_causal_lm_preprocessor import (
    GPTNeoXCausalLMPreprocessor,
)


@keras_nlp_export("keras_nlp.models.GPTNeoXCausalLM")
class GPTNeoXCausalLM(CausalLM):
    """An end-to-end GPTNeoX model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a GPT-NeoX model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    Args:
        backbone: A `keras_nlp.models.GPTNeoXBackbone` instance.
        preprocessor: A `keras_nlp.models.GPTNeoXCausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.
    """

    backbone_cls = GPTNeoXBackbone
    preprocessor_cls = GPTNeoXCausalLMPreprocessor

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
        """Build an empty cache for use with `call_with_cache()`."""
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        return ops.zeros(shape, dtype=self.compute_dtype)

    def call_with_cache(
        self,
        token_ids,
        cache,
        index,
    ):
        x = self.backbone.token_embedding(token_ids)
        x = self.backbone.embeddings_dropout(x)
        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=index,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        x = self.backbone.layer_norm(x)
        hidden_states = x
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states, cache
