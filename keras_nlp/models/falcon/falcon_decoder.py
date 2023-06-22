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

"""Falcon Decoder Layer"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LayerNormalization
from keras_nlp.models.falcon.falcon_mlp import FalconMLP
from keras_nlp.models.falcon.falcon_attention import FalconAttention

from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.FalconDecoder")
class FalconDecoder(keras.layers.Layer):
    """FalconDecoder used in Falcon models for decoding.

    This layer implements the decoder layer used in Falcon models. It consists of
    self-attention mechanism followed by a feed-forward neural network (MLP).

    Args:
        config: Configuration object containing the hyperparameters.

    Inputs:
        - hidden_states: Tensor of shape `(batch_size, seq_length, hidden_size)`.
          Input hidden states to the decoder layer.
        - alibi: Tensor of shape `(batch_size, num_heads, 1, kv_length)` containing
          alibi values for each head. Set to None if not using alibi.
        - attention_mask: Tensor of shape `(batch_size, 1, 1, seq_length)` containing
          the attention mask. Set to None if not using attention mask.
        - layer_past: Tuple of tensors `(past_key, past_value)` containing the cached
          key and value states from previous layers. Set to None if not using cache.
        - head_mask: Tensor of shape `(num_heads,)` containing the head mask. Set to None
          if not using head mask.
        - use_cache: Boolean value indicating whether to use the cache.
        - output_attentions: Boolean value indicating whether to output attention scores.
        - training: Boolean value indicating whether the layer is in training mode.

    Outputs:
        - outputs: Tuple containing the output tensor and additional outputs.
            - hidden_states: Tensor of shape `(batch_size, seq_length, hidden_size)`.
              Output hidden states from the decoder layer.
            - present: Tuple of tensors `(key_layer, value_layer)` containing the updated
              key and value states for caching.
            - attentions (optional): Tensor of shape `(batch_size, num_heads, seq_length, seq_length)`
              containing the attention scores. Only present if `output_attentions=True`.

    Examples:
        ```python
        config = DecoderLayerConfig(hidden_size=768, n_head=12, layer_norm_epsilon=1e-6,
                                    attention_dropout=0.1, hidden_dropout=0.1,
                                    apply_residual_connection_post_layernorm=True,
                                    parallel_attn=False)
        decoder_layer = FalconDecoder(config)
        outputs = decoder_layer(hidden_states, alibi, attention_mask, layer_past=None,
                                head_mask=None, use_cache=False, output_attentions=False,
                                training=True)
        ```
    """

    def __init__(self, config):
        super(FalconDecoder, self).__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = FalconAttention(config)

        if not config.parallel_attn:
            self.post_attention_layernorm = LayerNormalization(epsilon=config.layer_norm_epsilon)

        self.mlp = FalconMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

        self.config = config

    def call(
        self,
        hidden_states,
        alibi,
        attention_mask,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        training=None,
    ):
        """Call method of the FalconDecoderLayer.

        This method is called when applying the FalconDecoderLayer as a layer in a model.
        It performs the forward pass of the decoder layer.

        Args:
            hidden_states: Input hidden states to the decoder layer.
            alibi: Alibi values for each head.
            attention_mask: Attention mask.
            layer_past: Cached key and value states from previous layers.
            head_mask: Head mask.
            use_cache: Boolean value indicating whether to use the cache.
            output_attentions: Boolean value indicating whether to output attention scores.
            training: Boolean value indicating whether the layer is in training mode.

        Returns:
            outputs: Tuple containing the output tensor and additional outputs.

        """

        layernorm_output = self.input_layernorm(hidden_states)
        residual = hidden_states

        attn_outputs = self.self_attention(
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )

        attention_output = attn_outputs[0]

        if not self.config.parallel_attn:
            residual = tf.keras.layers.Dropout(self.config.attention_dropout)(residual, training=training)
            residual += attention_output
            layernorm_output = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        mlp_output = self.mlp(layernorm_output)

        if self.config.parallel_attn:
            mlp_output += attention_output

        output = tf.keras.layers.Dropout(self.config.hidden_dropout)(mlp_output, training=training)
        output += residual

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs
