import math

import keras
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import gelu_approximate
from keras_hub.src.utils.keras_utils import standardize_data_format


class AdaptiveLayerNormalization(layers.Layer):
    """Adaptive layer normalization.

    Args:
        embedding_dim: int. The size of each embedding vector.
        residual_modulation: bool. Whether to output the modulation parameters
            of the residual connection within the block of the diffusion
            transformers. Defaults to `False`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    References:
    - [FiLM: Visual Reasoning with a General Conditioning Layer](
    https://arxiv.org/abs/1709.07871).
    - [Scalable Diffusion Models with Transformers](
    https://arxiv.org/abs/2212.09748).
    """

    def __init__(self, hidden_dim, residual_modulation=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.residual_modulation = bool(residual_modulation)
        num_modulations = 6 if self.residual_modulation else 2

        self.silu = layers.Activation("silu", dtype=self.dtype_policy)
        self.dense = layers.Dense(
            num_modulations * hidden_dim, dtype=self.dtype_policy, name="dense"
        )
        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False,
            dtype="float32",
            name="norm",
        )

    def build(self, inputs_shape, embeddings_shape):
        self.silu.build(embeddings_shape)
        self.dense.build(embeddings_shape)
        self.norm.build(inputs_shape)

    def call(self, inputs, embeddings, training=None):
        x = inputs
        emb = self.dense(self.silu(embeddings), training=training)
        if self.residual_modulation:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                ops.split(emb, 6, axis=1)
            )
        else:
            shift_msa, scale_msa = ops.split(emb, 2, axis=1)
        scale_msa = ops.expand_dims(scale_msa, axis=1)
        shift_msa = ops.expand_dims(shift_msa, axis=1)
        x = ops.add(
            ops.multiply(
                self.norm(x, training=training),
                ops.add(1.0, scale_msa),
            ),
            shift_msa,
        )
        if self.residual_modulation:
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "residual_modulation": self.residual_modulation,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape, embeddings_shape):
        if self.residual_modulation:
            return (
                inputs_shape,
                embeddings_shape,
                embeddings_shape,
                embeddings_shape,
                embeddings_shape,
            )
        else:
            return inputs_shape


class MLP(layers.Layer):
    """A MLP block with architecture.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        output_dim: int. The number of units in the output layer.
        activation: str of callable. Activation to use in the hidden layers.
            Default to `None`.
    """

    def __init__(self, hidden_dim, output_dim, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.activation = keras.activations.get(activation)

        self.dense1 = layers.Dense(
            hidden_dim,
            activation=self.activation,
            dtype=self.dtype_policy,
            name="dense1",
        )
        self.dense2 = layers.Dense(
            output_dim,
            activation=None,
            dtype=self.dtype_policy,
            name="dense2",
        )

    def build(self, inputs_shape):
        self.dense1.build(inputs_shape)
        inputs_shape = self.dense1.compute_output_shape(inputs_shape)
        self.dense2.build(inputs_shape)

    def call(self, inputs, training=None):
        x = self.dense1(inputs, training=training)
        return self.dense2(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.output_dim
        return outputs_shape


class PatchEmbedding(layers.Layer):
    """A layer that converts images into patches.

    Args:
        patch_size: int. The size of one side of each patch.
        hidden_dim: int. The number of units in the hidden layers.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

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
    """A position embedding layer with adjustable height and width.

    The embedding will be cropped to match the input dimensions.

    Args:
        height: int. The maximum height of the embedding.
        width: int. The maximum width of the embedding.
    """

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

    def get_config(self):
        config = super().get_config()
        del config["sequence_length"]
        config.update(
            {
                "height": self.height,
                "width": self.width,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class TimestepEmbedding(layers.Layer):
    """A layer which learns embedding for input timesteps.

    Args:
        embedding_dim: int. The size of the embedding.
        frequency_dim: int. The size of the frequency.
        max_period: int. Controls the maximum frequency of the embeddings.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    Reference:
    - [Denoising Diffusion Probabilistic Models](
    https://arxiv.org/abs/2006.11239).
    """

    def __init__(
        self, embedding_dim, frequency_dim=256, max_period=10000, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = int(embedding_dim)
        self.frequency_dim = int(frequency_dim)
        self.max_period = float(max_period)
        # Precomputed `freq`.
        half_frequency_dim = frequency_dim // 2
        self.freq = ops.exp(
            ops.divide(
                ops.multiply(
                    -math.log(max_period),
                    ops.arange(0, half_frequency_dim, dtype="float32"),
                ),
                half_frequency_dim,
            )
        )

        self.mlp = MLP(
            embedding_dim,
            embedding_dim,
            "silu",
            dtype=self.dtype_policy,
            name="mlp",
        )

    def build(self, inputs_shape):
        embedding_shape = list(inputs_shape)[1:]
        embedding_shape.append(self.frequency_dim)
        self.mlp.build(embedding_shape)

    def _create_timestep_embedding(self, inputs):
        compute_dtype = keras.backend.result_type(self.compute_dtype, "float32")
        x = ops.cast(inputs, compute_dtype)
        freqs = ops.cast(self.freq, compute_dtype)
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
                "frequency_dim": self.frequency_dim,
                "max_period": self.max_period,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        output_shape = list(inputs_shape)[1:]
        output_shape.append(self.embedding_dim)
        return output_shape


class DismantledBlock(layers.Layer):
    """A dismantled block used to compute pre- and post-attention.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. The number of units in the hidden layers.
        mlp_ratio: float. The expansion ratio of `MLP`.
        use_projection: bool. Whether to use an attention projection layer at
            the end of the block.
        qk_norm: Optional str. Whether to normalize the query and key tensors.
            Available options are `None` and `"rms_norm"`. Defaults to `None`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_ratio=4.0,
        use_projection=True,
        qk_norm=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.use_projection = use_projection
        self.qk_norm = qk_norm

        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        if use_projection:
            self.ada_layer_norm = AdaptiveLayerNormalization(
                hidden_dim,
                residual_modulation=True,
                dtype=self.dtype_policy,
                name="ada_layer_norm",
            )
        else:
            self.ada_layer_norm = AdaptiveLayerNormalization(
                hidden_dim, dtype=self.dtype_policy, name="ada_layer_norm"
            )
        self.attention_qkv = layers.Dense(
            hidden_dim * 3, dtype=self.dtype_policy, name="attention_qkv"
        )
        if qk_norm is not None and qk_norm == "rms_norm":
            self.q_norm = layers.LayerNormalization(
                epsilon=1e-6, rms_scaling=True, dtype="float32", name="q_norm"
            )
            self.k_norm = layers.LayerNormalization(
                epsilon=1e-6, rms_scaling=True, dtype="float32", name="q_norm"
            )
        elif qk_norm is not None:
            raise NotImplementedError(
                "Supported `qk_norm` are `'rms_norm'` and `None`. "
                f"Received: qk_norm={qk_norm}."
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
            self.mlp = MLP(
                mlp_hidden_dim,
                hidden_dim,
                gelu_approximate,
                dtype=self.dtype_policy,
                name="mlp",
            )

    def build(self, inputs_shape, timestep_embedding):
        self.ada_layer_norm.build(inputs_shape, timestep_embedding)
        self.attention_qkv.build(inputs_shape)
        if self.qk_norm is not None:
            # [batch_size, sequence_length, num_heads, head_dim]
            self.q_norm.build([None, None, self.num_heads, self.head_dim])
            self.k_norm.build([None, None, self.num_heads, self.head_dim])
        if self.use_projection:
            self.attention_proj.build(inputs_shape)
            self.norm2.build(inputs_shape)
            self.mlp.build(inputs_shape)

    def _modulate(self, inputs, shift, scale):
        inputs = ops.cast(inputs, self.compute_dtype)
        shift = ops.cast(shift, self.compute_dtype)
        scale = ops.cast(scale, self.compute_dtype)
        return ops.add(ops.multiply(inputs, ops.add(scale, 1.0)), shift)

    def _compute_pre_attention(self, inputs, timestep_embedding, training=None):
        batch_size = ops.shape(inputs)[0]
        if self.use_projection:
            x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_layer_norm(
                inputs, timestep_embedding, training=training
            )
            qkv = self.attention_qkv(x, training=training)
            qkv = ops.reshape(
                qkv, (batch_size, -1, 3, self.num_heads, self.head_dim)
            )
            q, k, v = ops.unstack(qkv, 3, axis=2)
            if self.qk_norm is not None:
                q = self.q_norm(q, training=training)
                k = self.k_norm(k, training=training)
            return (q, k, v), (inputs, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            x = self.ada_layer_norm(
                inputs, timestep_embedding, training=training
            )
            qkv = self.attention_qkv(x, training=training)
            qkv = ops.reshape(
                qkv, (batch_size, -1, 3, self.num_heads, self.head_dim)
            )
            q, k, v = ops.unstack(qkv, 3, axis=2)
            if self.qk_norm is not None:
                q = self.q_norm(q, training=training)
                k = self.k_norm(k, training=training)
            return (q, k, v)

    def _compute_post_attention(
        self, inputs, inputs_intermediates, training=None
    ):
        x, gate_msa, shift_mlp, scale_mlp, gate_mlp = inputs_intermediates
        gate_msa = ops.expand_dims(gate_msa, axis=1)
        shift_mlp = ops.expand_dims(shift_mlp, axis=1)
        scale_mlp = ops.expand_dims(scale_mlp, axis=1)
        gate_mlp = ops.expand_dims(gate_mlp, axis=1)
        attn = self.attention_proj(inputs, training=training)
        x = ops.add(x, ops.multiply(gate_msa, attn))
        x = ops.add(
            x,
            ops.multiply(
                gate_mlp,
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
                "qk_norm": self.qk_norm,
            }
        )
        return config


class MMDiTBlock(layers.Layer):
    """A MMDiT block consisting of two `DismantledBlock` layers.

    One `DismantledBlock` processes the input latents, and the other processes
    the context embedding. This block integrates two modalities within the
    attention operation, allowing each representation to operate in its own
    space while considering the other.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. The number of units in the hidden layers.
        mlp_ratio: float. The expansion ratio of `MLP`.
        use_context_projection: bool. Whether to use an attention projection
            layer at the end of the context block.
        qk_norm: Optional str. Whether to normalize the query and key tensors.
            Available options are `None` and `"rms_norm"`. Defaults to `None`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    Reference:
    - [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](
    https://arxiv.org/abs/2403.03206)
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_ratio=4.0,
        use_context_projection=True,
        qk_norm=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.use_context_projection = use_context_projection
        self.qk_norm = qk_norm

        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self._inverse_sqrt_key_dim = 1.0 / math.sqrt(head_dim)

        self.x_block = DismantledBlock(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            use_projection=True,
            qk_norm=qk_norm,
            dtype=self.dtype_policy,
            name="x_block",
        )
        self.context_block = DismantledBlock(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            use_projection=use_context_projection,
            qk_norm=qk_norm,
            dtype=self.dtype_policy,
            name="context_block",
        )
        self.softmax = layers.Softmax(dtype="float32")

    def build(self, inputs_shape, context_shape, timestep_embedding_shape):
        self.x_block.build(inputs_shape, timestep_embedding_shape)
        self.context_block.build(context_shape, timestep_embedding_shape)

    def _compute_attention(self, query, key, value):
        batch_size = ops.shape(query)[0]

        # Use the fast path when `ops.dot_product_attention` and flash attention
        # are available.
        if hasattr(ops, "dot_product_attention") and hasattr(
            keras.config, "is_flash_attention_enabled"
        ):
            # `ops.dot_product_attention` is slower than the vanilla
            # implementation in the tensorflow backend.
            encoded = ops.dot_product_attention(
                query,
                key,
                value,
                scale=self._inverse_sqrt_key_dim,
                flash_attention=keras.config.is_flash_attention_enabled(),
            )
            return ops.reshape(
                encoded, (batch_size, -1, self.num_heads * self.head_dim)
            )

        # Ref: jax.nn.dot_product_attention
        # https://github.com/jax-ml/jax/blob/db89c245ac66911c98f265a05956fdfa4bc79d83/jax/_src/nn/functions.py#L846
        logits = ops.einsum("BTNH,BSNH->BNTS", query, key)
        logits = ops.multiply(logits, self._inverse_sqrt_key_dim)
        probs = self.softmax(logits)
        probs = ops.cast(probs, self.compute_dtype)
        encoded = ops.einsum("BNTS,BSNH->BTNH", probs, value)
        encoded = ops.reshape(
            encoded, (batch_size, -1, self.num_heads * self.head_dim)
        )
        return encoded

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
                "qk_norm": self.qk_norm,
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


class Unpatch(layers.Layer):
    """A layer that reconstructs the image from hidden patches.

    Args:
        patch_size: int. The size of each square patch in the input image.
        output_dim: int. The number of units in the output layer.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

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
    """A Multimodal Diffusion Transformer (MMDiT) model.

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
        qk_norm: Optional str. Whether to normalize the query and key tensors in
            the intermediate blocks. Available options are `None` and
            `"rms_norm"`. Defaults to `None`.
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
        qk_norm=None,
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
        self.vector_embedding = MLP(
            hidden_dim, hidden_dim, "silu", dtype=dtype, name="vector_embedding"
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
                qk_norm=qk_norm,
                dtype=dtype,
                name=f"joint_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.output_ada_layer_norm = AdaptiveLayerNormalization(
            hidden_dim, dtype=dtype, name="output_ada_layer_norm"
        )
        self.output_dense = layers.Dense(
            output_dim_in_final, dtype=dtype, name="output_dense"
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
        x = self.output_ada_layer_norm(x, timestep_embedding)
        x = self.output_dense(x)
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
        self.qk_norm = qk_norm

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
                "qk_norm": self.qk_norm,
            }
        )
        return config
