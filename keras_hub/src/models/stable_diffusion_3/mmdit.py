import math

import keras
from keras import layers
from keras import models
from keras import ops

from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import gelu_approximate
from keras_hub.src.utils.keras_utils import standardize_data_format


class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, hidden_dim, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        data_format = standardize_data_format(data_format)

        self.patch_embedding = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="patch_embedding",
        )

    def build(self, input_shape):
        self.patch_embedding.build(input_shape)

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        x_shape = ops.shape(x)
        x = ops.reshape(x, (x_shape[0], x_shape[1] * x_shape[2], x_shape[3]))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


class AdjustablePositionEmbedding(PositionEmbedding):
    def __init__(
        self,
        height,
        width,
        initializer="glorot_uniform",
        **kwargs,
    ):
        height = int(height)
        width = int(width)
        sequence_length = height * width
        super().__init__(sequence_length, initializer, **kwargs)
        self.height = height
        self.width = width

    def call(self, inputs, height=None, width=None):
        height = height or self.height
        width = width or self.width
        shape = ops.shape(inputs)
        feature_length = shape[-1]
        top = ops.cast(ops.floor_divide(self.height - height, 2), "int32")
        left = ops.cast(ops.floor_divide(self.width - width, 2), "int32")
        position_embedding = ops.convert_to_tensor(self.position_embeddings)
        position_embedding = ops.reshape(
            position_embedding, (self.height, self.width, feature_length)
        )
        position_embedding = ops.slice(
            position_embedding,
            (top, left, 0),
            (height, width, feature_length),
        )
        position_embedding = ops.reshape(
            position_embedding, (height * width, feature_length)
        )
        position_embedding = ops.expand_dims(position_embedding, axis=0)
        return position_embedding

    def compute_output_shape(self, input_shape):
        return input_shape


class TimestepEmbedding(layers.Layer):
    def __init__(
        self, embedding_dim, frequency_dim=256, max_period=10000, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = int(embedding_dim)
        self.frequency_dim = int(frequency_dim)
        self.max_period = float(max_period)
        self.half_frequency_dim = self.frequency_dim // 2

        self.mlp = models.Sequential(
            [
                layers.Dense(
                    embedding_dim, activation="silu", dtype=self.dtype_policy
                ),
                layers.Dense(
                    embedding_dim, activation=None, dtype=self.dtype_policy
                ),
            ],
            name="mlp",
        )

    def build(self, inputs_shape):
        embedding_shape = list(inputs_shape)[1:]
        embedding_shape.append(self.frequency_dim)
        self.mlp.build(embedding_shape)

    def _create_timestep_embedding(self, inputs):
        compute_dtype = keras.backend.result_type(self.compute_dtype, "float32")
        x = ops.cast(inputs, compute_dtype)
        freqs = ops.exp(
            ops.divide(
                ops.multiply(
                    -math.log(self.max_period),
                    ops.arange(0, self.half_frequency_dim, dtype="float32"),
                ),
                self.half_frequency_dim,
            )
        )
        freqs = ops.cast(freqs, compute_dtype)
        x = ops.multiply(x, ops.expand_dims(freqs, axis=0))
        embedding = ops.concatenate([ops.cos(x), ops.sin(x)], axis=-1)
        if self.frequency_dim % 2 != 0:
            embedding = ops.pad(embedding, [[0, 0], [0, 1]])
        return ops.cast(embedding, self.compute_dtype)

    def call(self, inputs, training=None):
        timestep_embedding = self._create_timestep_embedding(inputs)
        return self.mlp(timestep_embedding, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_period": self.max_period,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        output_shape = list(inputs_shape)[1:]
        output_shape.append(self.embedding_dim)
        return output_shape


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
            dtype="float32",
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
                dtype="float32",
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
        self.softmax = layers.Softmax(dtype="float32")

    def build(self, inputs_shape, context_shape, timestep_embedding_shape):
        self.x_block.build(inputs_shape, timestep_embedding_shape)
        self.context_block.build(context_shape, timestep_embedding_shape)

    def _compute_attention(self, query, key, value):
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )
        attention_scores = ops.einsum(self._dot_product_equation, key, query)
        attention_scores = self.softmax(attention_scores)
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
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


class OutputLayer(layers.Layer):
    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        num_modulation = 2

        self.adaptive_norm_modulation = models.Sequential(
            [
                layers.Activation("silu", dtype=self.dtype_policy),
                layers.Dense(
                    num_modulation * hidden_dim, dtype=self.dtype_policy
                ),
            ],
            name="adaptive_norm_modulation",
        )
        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False,
            dtype="float32",
            name="norm",
        )
        self.output_dense = layers.Dense(
            output_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="output_dense",
        )

    def build(self, inputs_shape, timestep_embedding_shape):
        self.adaptive_norm_modulation.build(timestep_embedding_shape)
        self.norm.build(inputs_shape)
        self.output_dense.build(inputs_shape)

    def _modulate(self, inputs, shift, scale):
        shift = ops.expand_dims(shift, axis=1)
        scale = ops.expand_dims(scale, axis=1)
        return ops.add(ops.multiply(inputs, ops.add(scale, 1.0)), shift)

    def call(self, inputs, timestep_embedding, training=None):
        x = inputs
        modulation = self.adaptive_norm_modulation(
            timestep_embedding, training=training
        )
        modulation = ops.reshape(modulation, (-1, 2, self.hidden_dim))
        shift, scale = ops.unstack(modulation, 2, axis=1)
        x = self._modulate(self.norm(x), shift, scale)
        x = self.output_dense(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.output_dim
        return outputs_shape


class Unpatch(layers.Layer):
    def __init__(self, patch_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.output_dim = int(output_dim)

    def call(self, inputs, height, width):
        patch_size = self.patch_size
        output_dim = self.output_dim
        x = ops.reshape(
            inputs,
            (-1, height, width, patch_size, patch_size, output_dim),
        )
        # (b, h, w, p1, p2, o) -> (b, h, p1, w, p2, o)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return ops.reshape(
            x,
            (-1, height * patch_size, width * patch_size, output_dim),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "output_dim": self.output_dim,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape)
        return [inputs_shape[0], None, None, self.output_dim]


class MMDiT(Backbone):
    """Multimodal Diffusion Transformer (MMDiT) model for Stable Diffusion 3.

    MMDiT is introduced in [
    Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](
    https://arxiv.org/abs/2403.03206).

    Args:
        patch_size: int. The size of each square patch in the input image.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        position_size: int. The size of the height and width for the position
            embedding.
        mlp_ratio: float. The ratio of the mlp hidden dim to the transformer
        latent_shape: tuple. The shape of the latent image.
        context_shape: tuple. The shape of the context.
        pooled_projection_shape: tuple. The shape of the pooled projection.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.
    """

    def __init__(
        self,
        patch_size,
        hidden_dim,
        num_layers,
        num_heads,
        position_size,
        mlp_ratio=4.0,
        latent_shape=(64, 64, 16),
        context_shape=(None, 4096),
        pooled_projection_shape=(2048,),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if None in latent_shape:
            raise ValueError(
                "`latent_shape` must be fully specified. "
                f"Received: latent_shape={latent_shape}"
            )
        image_height = latent_shape[0] // patch_size
        image_width = latent_shape[1] // patch_size
        output_dim = latent_shape[-1]
        output_dim_in_final = patch_size**2 * output_dim
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError(
                "Currently only 'channels_last' is supported."
            )

        # === Layers ===
        self.patch_embedding = PatchEmbedding(
            patch_size,
            hidden_dim,
            data_format=data_format,
            dtype=dtype,
            name="patch_embedding",
        )
        self.position_embedding_add = layers.Add(
            dtype=dtype, name="position_embedding_add"
        )
        self.position_embedding = AdjustablePositionEmbedding(
            position_size, position_size, dtype=dtype, name="position_embedding"
        )
        self.context_embedding = layers.Dense(
            hidden_dim,
            dtype=dtype,
            name="context_embedding",
        )
        self.vector_embedding = models.Sequential(
            [
                layers.Dense(hidden_dim, activation="silu", dtype=dtype),
                layers.Dense(hidden_dim, activation=None, dtype=dtype),
            ],
            name="vector_embedding",
        )
        self.vector_embedding_add = layers.Add(
            dtype=dtype, name="vector_embedding_add"
        )
        self.timestep_embedding = TimestepEmbedding(
            hidden_dim, dtype=dtype, name="timestep_embedding"
        )
        self.joint_blocks = [
            MMDiTBlock(
                num_heads,
                hidden_dim,
                mlp_ratio,
                use_context_projection=not (i == num_layers - 1),
                dtype=dtype,
                name=f"joint_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.output_layer = OutputLayer(
            hidden_dim, output_dim_in_final, dtype=dtype, name="output_layer"
        )
        self.unpatch = Unpatch(
            patch_size, output_dim, dtype=dtype, name="unpatch"
        )

        # === Functional Model ===
        latent_inputs = layers.Input(shape=latent_shape, name="latent")
        context_inputs = layers.Input(shape=context_shape, name="context")
        pooled_projection_inputs = layers.Input(
            shape=pooled_projection_shape, name="pooled_projection"
        )
        timestep_inputs = layers.Input(shape=(1,), name="timestep")

        # Embeddings.
        x = self.patch_embedding(latent_inputs)
        position_embedding = self.position_embedding(
            x, height=image_height, width=image_width
        )
        x = self.position_embedding_add([x, position_embedding])
        context = self.context_embedding(context_inputs)
        pooled_projection = self.vector_embedding(pooled_projection_inputs)
        timestep_embedding = self.timestep_embedding(timestep_inputs)
        timestep_embedding = self.vector_embedding_add(
            [timestep_embedding, pooled_projection]
        )

        # Blocks.
        for block in self.joint_blocks:
            if block.use_context_projection:
                x, context = block(x, context, timestep_embedding)
            else:
                x = block(x, context, timestep_embedding)

        # Output layer.
        x = self.output_layer(x, timestep_embedding)
        outputs = self.unpatch(x, height=image_height, width=image_width)

        super().__init__(
            inputs={
                "latent": latent_inputs,
                "context": context_inputs,
                "pooled_projection": pooled_projection_inputs,
                "timestep": timestep_inputs,
            },
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.position_size = position_size
        self.mlp_ratio = mlp_ratio
        self.latent_shape = latent_shape
        self.context_shape = context_shape
        self.pooled_projection_shape = pooled_projection_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "position_size": self.position_size,
                "mlp_ratio": self.mlp_ratio,
                "latent_shape": self.latent_shape,
                "context_shape": self.context_shape,
                "pooled_projection_shape": self.pooled_projection_shape,
            }
        )
        return config
