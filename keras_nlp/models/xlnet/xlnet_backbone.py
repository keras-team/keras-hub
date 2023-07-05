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
"""XLNet backbone model."""


import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.xlnet.xlnet_content_and_query_embedding import (
    ContentAndQueryEmbedding,
)
from keras_nlp.models.xlnet.xlnet_encoder import XLNetEncoder


def cache_mem(curr_out, prev_mem):
    if prev_mem is None:
        new_mem = curr_out
    else:
        new_mem = tf.concat([prev_mem, curr_out], 1)

    return tf.stop_gradient(new_mem)


@keras_nlp_export("keras_nlp.models.XLNetBackbone")
class XLNetBackbone(Backbone):
    """XLNet encoder network.

    This class implements a XLNet Transformer.

    The default constructor gives a fully customizable, randomly initialized XLNet
    encoder with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Attributes:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer encoder layers.
        num_heads: int, the number of heads in the
            `keras.layers.TwoStreamRelativeAttention` layer.
        hidden_dim: int, the size hidden states.
        intermediate_dim: int, the hidden size of feedforward network.
        dropout: float, defaults to 0.0 the dropout value, shared by
            `keras.layers.TwoStreamRelativeAttention` and feedforward network.
        activation: string or `keras.activations`, defaults to "gelu". the
            activation function of feedforward network.
        kernel_initializer_range: int, defaults to 0.02. The kernel initializer
            range for the dense and relative attention layers.
        bias_initializer: string or `keras.initializers` initializer,
            defaults to "zeros". The bias initializer for
            the dense and multiheaded relative attention layers.
        layer_norm_epsilon: float, defaults to 1e-12. The epsilon value in layer
            normalization components.
        **kwargs: other keyword arguments.

    Call Args:
        token_ids: Indices of input sequence tokens in the vocabulary of shape `[batch_size, sequence_length]`.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs of shape
            `[batch_size, sequence_length]`.
        padding_mask: Mask to avoid performing attention on padding token indices of shape `[batch_size, sequence_length]`.
        target_mapping: Optional `Tensor` which denotes mask to indicate the output tokens to use of shape
            `[batch_size, num_predict, sequence_length]`. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k
            is on the j-th token. Only used during pretraining.
        perm_mask: Optional `Tensor` which denotes mask to indicate the attention pattern for each input token of shape
            `[batch_size, sequence_length, sequence_length]`. Only used during pretraining.
        mems: Optional `Tensor` of shape `[batch_size, sequence_length(of previous hidden state), hidden_dim, num_layers]`
            to denote the previous hidden states. If passed, this is also attended over as in Transformer-XL.

    Returns:
        last_hidden_state: last hidden state of query state of shape `[batch_size, num_predict, hidden_dim]` if query state is not None
            otherwise last hidden state of content of shape `[batch_size, sequence_length, hidden_dim]`.
        new_mems: new memory units returned by the model. These are the conatenated
            tensors of previous mems and hidden states of most recent pass.

    Examples:
    ```python
    input_data = {
        "token_ids": tf.constant(
            [460, 5272, 1758, 4905, 9, 4, 3], shape=(1, 7),
        ),
        "token_type_ids": tf.constant(
            [0, 0, 0, 0, 0, 0, 2], shape=(1, 7),
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1], shape=(1, 7)
        ),
        "target_mapping": tf.random.uniform(minval=0, maxval=2, shape=(1, 5, 7), dtype=tf.int64),
        "perm_mask": tf.random.uniform(minval=0, maxval=2, shape=(1, 7, 7), dtype=tf.int64),
        "mems": tf.random.uniform((12, 7, 1, 768), dtype=tf.float64)
    }

    # Randomly initialized XLNet encoder with a custom config
    model = keras_nlp.models.XLNetBackbone(
        vocabulary_size=32000,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.0,
        activation="gelu",
        kernel_initializer_range=0.02,
        bias_initializer="zeros",
        layer_norm_epsilon=1e-12,
        **kwargs,
    ):
        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        token_type_id = keras.Input(
            shape=(None,), dtype="int32", name="token_type_ids"
        )
        mems = tf.keras.Input(
            shape=(None, None, None), dtype="float32", name="mems"
        )

        # Only used during pretraining
        perm_mask = keras.Input(
            shape=(
                None,
                None,
            ),
            dtype="int32",
            name="perm_mask",
        )
        target_mapping = keras.Input(
            shape=(None, None), dtype="int32", name="target_mapping"
        )

        # Content and Query Embedding
        embedding_outputs = ContentAndQueryEmbedding(
            vocabulary_size=vocabulary_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_initializer_range=kernel_initializer_range,
            name="content_query_embedding",
        )(
            token_id_input=token_id_input,
            perm_mask=perm_mask,
            token_type_id=token_type_id,
            mems=mems,
            padding_mask=padding_mask,
            target_mapping=target_mapping,
        )
        (
            output_h,
            output_g,
            pos_emb,
            tgt_map,
            seg_mat,
            mems_opt,
            attn_mask_h,
            attn_mask_g,
        ) = embedding_outputs

        # Encoders
        new_mems = ()
        head_dim = hidden_dim // num_heads
        for i in range(num_layers):
            new_mems = new_mems + (cache_mem(output_h, mems_opt[:, :, :, i]),)

            output_h, output_g = XLNetEncoder(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                kernel_initializer_range=kernel_initializer_range,
                bias_initializer=bias_initializer,
                name=f"xlnet_encoder_{i}",
            )(
                output_h=output_h,
                output_g=output_g,
                pos_emb=pos_emb,
                target_mapping=tgt_map,
                seg_mat=seg_mat,
                mems=mems_opt[:, :, :, i],
                attn_mask_h=attn_mask_h,
                attn_mask_g=attn_mask_g,
            )

        output = keras.layers.Dropout(dropout)(
            output_g if output_g is not None else output_h
        )

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask,
                "token_type_ids": token_type_id,
                "mems": mems,
                "perm_mask": perm_mask,
                "target_mapping": target_mapping,
            },
            outputs={"last_hidden_state": output, "new_mems": new_mems},
            **kwargs,
        )

        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = activation
        self.kernel_initializer_range = kernel_initializer_range
        self.bias_initializer = bias_initializer
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.layer_norm_epsilon,
                "activation": self.activation,
                "kernel_initializer_range": self.kernel_initializer_range,
                "bias_initializer": self.bias_initializer,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("content_query_embedding").word_embed
