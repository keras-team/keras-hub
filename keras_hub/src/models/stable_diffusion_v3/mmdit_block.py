# Copyright 2024 The KerasHub Authors
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
import math

from keras import layers
from keras import models
from keras import ops

from keras_hub.src.utils.keras_utils import gelu_approximate


class DismantledBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_ratio=4.0,
        use_projection=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.use_projection = use_projection

        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        num_modulations = 6 if use_projection else 2
        self.num_modulations = num_modulations

        self.adaptive_norm_modulation = models.Sequential(
            [
                layers.Activation("silu", dtype=self.dtype_policy),
                layers.Dense(
                    num_modulations * hidden_dim, dtype=self.dtype_policy
                ),
            ],
            name="adaptive_norm_modulation",
        )
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False,
            dtype=self.dtype_policy,
            name="norm1",
        )
        self.attention_qkv = layers.Dense(
            hidden_dim * 3, dtype=self.dtype_policy, name="attention_qkv"
        )
        if use_projection:
            self.attention_proj = layers.Dense(
                hidden_dim, dtype=self.dtype_policy, name="attention_proj"
            )
            self.norm2 = layers.LayerNormalization(
                epsilon=1e-6,
                center=False,
                scale=False,
                dtype=self.dtype_policy,
                name="norm2",
            )
            self.mlp = models.Sequential(
                [
                    layers.Dense(
                        mlp_hidden_dim,
                        activation=gelu_approximate,
                        dtype=self.dtype_policy,
                    ),
                    layers.Dense(
                        hidden_dim,
                        dtype=self.dtype_policy,
                    ),
                ],
                name="mlp",
            )

    def build(self, inputs_shape, timestep_embedding):
        self.adaptive_norm_modulation.build(timestep_embedding)
        self.attention_qkv.build(inputs_shape)
        self.norm1.build(inputs_shape)
        if self.use_projection:
            self.attention_proj.build(inputs_shape)
            self.norm2.build(inputs_shape)
            self.mlp.build(inputs_shape)

    def _modulate(self, inputs, shift, scale):
        shift = ops.expand_dims(shift, axis=1)
        scale = ops.expand_dims(scale, axis=1)
        return ops.add(ops.multiply(inputs, ops.add(scale, 1.0)), shift)

    def _compute_pre_attention(self, inputs, timestep_embedding, training=None):
        batch_size = ops.shape(inputs)[0]
        if self.use_projection:
            modulation = self.adaptive_norm_modulation(
                timestep_embedding, training=training
            )
            modulation = ops.reshape(
                modulation, (batch_size, 6, self.hidden_dim)
            )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = ops.unstack(modulation, 6, axis=1)
            qkv = self.attention_qkv(
                self._modulate(self.norm1(inputs), shift_msa, scale_msa),
                training=training,
            )
            qkv = ops.reshape(
                qkv, (batch_size, -1, 3, self.num_heads, self.head_dim)
            )
            q, k, v = ops.unstack(qkv, 3, axis=2)
            return (q, k, v), (inputs, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            modulation = self.adaptive_norm_modulation(
                timestep_embedding, training=training
            )
            modulation = ops.reshape(
                modulation, (batch_size, 2, self.hidden_dim)
            )
            shift_msa, scale_msa = ops.unstack(modulation, 2, axis=1)
            qkv = self.attention_qkv(
                self._modulate(self.norm1(inputs), shift_msa, scale_msa),
                training=training,
            )
            qkv = ops.reshape(
                qkv, (batch_size, -1, 3, self.num_heads, self.head_dim)
            )
            q, k, v = ops.unstack(qkv, 3, axis=2)
            return (q, k, v)

    def _compute_post_attention(
        self, inputs, inputs_intermediates, training=None
    ):
        x, gate_msa, shift_mlp, scale_mlp, gate_mlp = inputs_intermediates
        attn = self.attention_proj(inputs, training=training)
        x = ops.add(x, ops.multiply(ops.expand_dims(gate_msa, axis=1), attn))
        x = ops.add(
            x,
            ops.multiply(
                ops.expand_dims(gate_mlp, axis=1),
                self.mlp(
                    self._modulate(self.norm2(x), shift_mlp, scale_mlp),
                    training=training,
                ),
            ),
        )
        return x

    def call(
        self,
        inputs,
        timestep_embedding=None,
        inputs_intermediates=None,
        pre_attention=True,
        training=None,
    ):
        if pre_attention:
            return self._compute_pre_attention(
                inputs, timestep_embedding, training=training
            )
        else:
            return self._compute_post_attention(
                inputs, inputs_intermediates, training=training
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "mlp_ratio": self.mlp_ratio,
                "use_projection": self.use_projection,
            }
        )
        return config


class MMDiTBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_ratio=4.0,
        use_context_projection=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.use_context_projection = use_context_projection

        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self._inverse_sqrt_key_dim = 1.0 / math.sqrt(head_dim)
        self._dot_product_equation = "aecd,abcd->acbe"
        self._combine_equation = "acbe,aecd->abcd"

        self.x_block = DismantledBlock(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            use_projection=True,
            dtype=self.dtype_policy,
            name="x_block",
        )
        self.context_block = DismantledBlock(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            use_projection=use_context_projection,
            dtype=self.dtype_policy,
            name="context_block",
        )

    def build(self, inputs_shape, context_shape, timestep_embedding_shape):
        self.x_block.build(inputs_shape, timestep_embedding_shape)
        self.context_block.build(context_shape, timestep_embedding_shape)

    def _compute_attention(self, query, key, value):
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )
        attention_scores = ops.einsum(self._dot_product_equation, key, query)
        attention_scores = ops.nn.softmax(attention_scores, axis=-1)
        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )
        batch_size = ops.shape(attention_output)[0]
        attention_output = ops.reshape(
            attention_output, (batch_size, -1, self.num_heads * self.head_dim)
        )
        return attention_output

    def call(self, inputs, context, timestep_embedding, training=None):
        # Compute pre-attention.
        x = inputs
        if self.use_context_projection:
            context_qkv, context_intermediates = self.context_block(
                context,
                timestep_embedding=timestep_embedding,
                training=training,
            )
        else:
            context_qkv = self.context_block(
                context,
                timestep_embedding=timestep_embedding,
                training=training,
            )
        context_len = ops.shape(context_qkv[0])[1]
        x_qkv, x_intermediates = self.x_block(
            x, timestep_embedding=timestep_embedding, training=training
        )
        q = ops.concatenate([context_qkv[0], x_qkv[0]], axis=1)
        k = ops.concatenate([context_qkv[1], x_qkv[1]], axis=1)
        v = ops.concatenate([context_qkv[2], x_qkv[2]], axis=1)

        # Compute attention.
        attention = self._compute_attention(q, k, v)
        context_attention = attention[:, :context_len]
        x_attention = attention[:, context_len:]

        # Compute post-attention.
        x = self.x_block(
            x_attention,
            inputs_intermediates=x_intermediates,
            pre_attention=False,
            training=training,
        )
        if self.use_context_projection:
            context = self.context_block(
                context_attention,
                inputs_intermediates=context_intermediates,
                pre_attention=False,
                training=training,
            )
            return x, context
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "mlp_ratio": self.mlp_ratio,
                "use_context_projection": self.use_context_projection,
            }
        )
        return config

    def compute_output_shape(
        self, inputs_shape, context_shape, timestep_embedding_shape
    ):
        if self.use_context_projection:
            return inputs_shape, context_shape
        else:
            return inputs_shape
