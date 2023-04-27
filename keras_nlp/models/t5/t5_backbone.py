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

"""T5 backbone model."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.transformer_layer_utils import compute_causal_mask
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.t5.t5_layer_norm import T5LayerNorm
from keras_nlp.models.t5.t5_transformer_layer import T5TransformerLayer
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class T5Backbone(Backbone):
    """T5 encoder-decoder backbone model.

    T5 is a LLM pretrained on a mix of unsupervised and supervised tasks,
    where each task is converted to a sequence-to-sequence format.
    T5 works well on a variety of tasks out-of-the-box by prepending
    various prefixex to the input sequence, e.g., for translation:
    `"translate English to German: ..."`, for summarization:
    `"summarize: ..."`.

    T5 was introduced in
    [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

    The default constructor gives a fully customizable, randomly initialized T5
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of Transformer layers.
        num_heads: int. The number of attention heads for each Transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The hidden size of the Transformer layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each Transformer layer.
        dropout: float. Dropout probability for the Transformer layers.
        activation: activation function (or activation string name). The
            activation to be used in the inner dense blocks of the
            Transformer layers. Defaults to `"gelu"`. The original
            T5 architecture used `"relu"`,
            but more recent versions use `"gelu"`.
        use_gated_activation: boolean. Whether to use activation gating in
            the inner dense blocks of the Transformer layers.
            Defaults to True. The original T5 architecture didn't use
            gating, but more recent versions do.
        layer_norm_epsilon: float. Epsilon factor to be used in the
            layer normalization layers in the Transformer layers.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        activation="gelu",
        use_gated_activation=True,
        layer_norm_epsilon=1e-06,
        **kwargs,
    ):
        # Encoder inputs
        encoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        encoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )

        # Decoder inputs.
        decoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )

        # Token embedding layer. This layer is shared by encoder and decoder.
        token_embedding_layer = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(1.0),
            name="token_embedding",
        )

        # ===== Encoder =====

        # Embed tokens.
        token_embedding = token_embedding_layer(encoder_token_ids)
        x = keras.layers.Dropout(
            dropout,
            name="encoder_embedding_dropout",
        )(token_embedding)

        # Encoder attention mask is just our padding mask.
        encoder_attention_mask = encoder_padding_mask[:, tf.newaxis, :]

        position_bias = None
        for i in range(num_layers):
            x, position_bias = T5TransformerLayer(
                is_decoder=False,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                num_heads=num_heads,
                use_gated_activation=use_gated_activation,
                use_relative_attention_bias=bool(i == 0),
                name=f"transformer_encoder_layer_{i}",
            )(
                x,
                attention_mask=encoder_attention_mask,
                position_bias=position_bias,
            )

        x = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            name="encoder_output_layer_norm",
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="encoder_output_dropout",
        )(x)
        encoder_output = x

        # ===== Decoder =====

        # Embed tokens.
        token_embedding = token_embedding_layer(decoder_token_ids)
        x = keras.layers.Dropout(
            dropout,
            name="decoder_embedding_dropout",
        )(token_embedding)

        # Decoder attention mask is padding mask plus a causal mask.
        decoder_attention_mask = decoder_padding_mask[:, tf.newaxis, :]
        batch_size, length = tf.shape(x)[0], tf.shape(x)[1]
        causal_mask = compute_causal_mask(batch_size, length, length)
        decoder_attention_mask = causal_mask & decoder_attention_mask

        position_bias = None
        for i in range(num_layers):
            x, position_bias = T5TransformerLayer(
                is_decoder=True,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                num_heads=num_heads,
                use_gated_activation=use_gated_activation,
                use_relative_attention_bias=bool(i == 0),
                name=f"transformer_decoder_layer_{i}",
            )(
                x,
                attention_mask=decoder_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_attention_mask,
            )

        x = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            name="decoder_output_layer_norm",
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="decoder_output_dropout",
        )(x)
        decoder_output = x

        super().__init__(
            {
                "encoder_token_ids": encoder_token_ids,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_token_ids": decoder_token_ids,
                "decoder_padding_mask": decoder_padding_mask,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
            **kwargs,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "activation": keras.activations.serialize(self.activation),
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")

    @classproperty
    def presets(cls):
        return {}
