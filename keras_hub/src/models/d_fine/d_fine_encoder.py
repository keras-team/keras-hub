import keras
import numpy as np

from keras_hub.src.models.d_fine.d_fine_attention import DFineMultiheadAttention
from keras_hub.src.utils.keras_utils import clone_initializer


class DFineEncoderLayer(keras.layers.Layer):
    """Single encoder layer for D-FINE models.

    This layer is the fundamental building block of the `DFineEncoder`. It
    implements a standard transformer encoder layer with multi-head
    self-attention (`DFineMultiheadAttention`) and a feed-forward network. It is
    used to process and refine the feature sequences from the CNN backbone.

    Args:
        normalize_before: bool, Whether to apply layer normalization before
            the attention and feed-forward sub-layers (pre-norm) or after
            (post-norm).
        encoder_hidden_dim: int, Hidden dimension size of the encoder.
        num_attention_heads: int, Number of attention heads in multi-head
            attention.
        dropout: float, Dropout probability applied to attention outputs and
            feed-forward outputs.
        layer_norm_eps: float, Small constant added to the denominator for
            numerical stability in layer normalization.
        encoder_activation_function: str, Activation function used in the
            feed-forward network.
        activation_dropout: float, Dropout probability applied after the
            activation function in the feed-forward network.
        encoder_ffn_dim: int, Hidden dimension size of the feed-forward network.
        **kwargs: Additional keyword arguments passed to the parent class.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
    """

    def __init__(
        self,
        normalize_before,
        encoder_hidden_dim,
        num_attention_heads,
        dropout,
        layer_norm_eps,
        encoder_activation_function,
        activation_dropout,
        encoder_ffn_dim,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.normalize_before = normalize_before
        self.encoder_hidden_dim = encoder_hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps
        self.encoder_activation_function = encoder_activation_function
        self.activation_dropout_rate = activation_dropout
        self.encoder_ffn_dim = encoder_ffn_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.self_attn = DFineMultiheadAttention(
            embedding_dim=self.encoder_hidden_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout_rate,
            dtype=self.dtype_policy,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="self_attn",
        )
        self.self_attn_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="self_attn_layer_norm",
            dtype=self.dtype_policy,
        )
        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout_layer",
            dtype=self.dtype_policy,
        )
        self.activation_fn_layer = keras.layers.Activation(
            self.encoder_activation_function,
            name="activation_fn_layer",
            dtype=self.dtype_policy,
        )
        self.activation_dropout_layer = keras.layers.Dropout(
            rate=self.activation_dropout_rate,
            name="activation_dropout_layer",
            dtype=self.dtype_policy,
        )
        self.fc1 = keras.layers.Dense(
            self.encoder_ffn_dim,
            name="fc1",
            dtype=self.dtype_policy,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self.fc2 = keras.layers.Dense(
            self.encoder_hidden_dim,
            name="fc2",
            dtype=self.dtype_policy,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self.final_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="final_layer_norm",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.self_attn.build(input_shape)
        self.self_attn_layer_norm.build(input_shape)
        self.fc1.build(input_shape)
        self.fc2.build((input_shape[0], input_shape[1], self.encoder_ffn_dim))
        self.final_layer_norm.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        output_attentions=False,
        training=None,
    ):
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(
                hidden_states, training=training
            )
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = self.dropout_layer(hidden_states, training=training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(
                hidden_states, training=training
            )
        if self.normalize_before:
            hidden_states = self.final_layer_norm(
                hidden_states, training=training
            )
        residual_ffn = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn_layer(hidden_states)
        hidden_states = self.activation_dropout_layer(
            hidden_states, training=training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, training=training)
        hidden_states = residual_ffn + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(
                hidden_states, training=training
            )
        if training:
            dtype_name = keras.backend.standardize_dtype(self.compute_dtype)
            if dtype_name == "float16":
                clamp_value = np.finfo(np.float16).max - 1000.0
            else:  # float32, bfloat16
                clamp_value = np.finfo(np.float32).max - 1000.0
            hidden_states = keras.ops.clip(
                hidden_states, x_min=-clamp_value, x_max=clamp_value
            )
        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        output_attentions=False,
        training=None,
    ):
        attn_output_spec = self.self_attn.compute_output_spec(
            hidden_states,
            position_embeddings,
            attention_mask,
            output_attentions,
        )
        if output_attentions:
            hidden_states_output_spec, self_attn_weights_spec = attn_output_spec
            return hidden_states_output_spec, self_attn_weights_spec
        return attn_output_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "normalize_before": self.normalize_before,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "num_attention_heads": self.num_attention_heads,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
                "encoder_activation_function": self.encoder_activation_function,
                "activation_dropout": self.activation_dropout_rate,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config


class DFineEncoder(keras.layers.Layer):
    """Multi-layer encoder for D-FINE models.

    This layer implements a stack of `DFineEncoderLayer` instances. It is used
    within the `DFineHybridEncoder` to apply transformer-based processing to
    the feature maps from the CNN backbone, creating rich contextual
    representations before they are passed to the FPN/PAN pathways.

    Args:
        normalize_before: bool, Whether to apply layer normalization before
            the attention and feed-forward sub-layers (pre-norm) or after
            (post-norm) in each encoder layer.
        encoder_hidden_dim: int, Hidden dimension size of the encoder layers.
        num_attention_heads: int, Number of attention heads in multi-head
            attention for each layer.
        dropout: float, Dropout probability applied to attention outputs and
            feed-forward outputs in each layer.
        layer_norm_eps: float, Small constant added to the denominator for
            numerical stability in layer normalization.
        encoder_activation_function: str, Activation function used in the
            feed-forward networks of each layer.
        activation_dropout: float, Dropout probability applied after the
            activation function in the feed-forward networks.
        encoder_ffn_dim: int, Hidden dimension size of the feed-forward
            networks in each layer.
        num_encoder_layers: int, Number of encoder layers in the stack.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights of each layer. Defaults to
            `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights of each layer. Defaults to
            `"zeros"`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        normalize_before,
        encoder_hidden_dim,
        num_attention_heads,
        dropout,
        layer_norm_eps,
        encoder_activation_function,
        activation_dropout,
        encoder_ffn_dim,
        num_encoder_layers,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.normalize_before = normalize_before
        self.encoder_hidden_dim = encoder_hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps
        self.encoder_activation_function = encoder_activation_function
        self.activation_dropout_rate = activation_dropout
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_encoder_layers = num_encoder_layers
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.encoder_layer = []
        for i in range(self.num_encoder_layers):
            layer = DFineEncoderLayer(
                normalize_before=self.normalize_before,
                encoder_hidden_dim=self.encoder_hidden_dim,
                num_attention_heads=self.num_attention_heads,
                dropout=self.dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                encoder_activation_function=self.encoder_activation_function,
                activation_dropout=self.activation_dropout_rate,
                encoder_ffn_dim=self.encoder_ffn_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
                name=f"encoder_layer_{i}",
            )
            self.encoder_layer.append(layer)

    def build(self, input_shape):
        current_input_shape_for_layer = input_shape
        for encoder_layer_instance in self.encoder_layer:
            encoder_layer_instance.build(current_input_shape_for_layer)
        super().build(input_shape)

    def compute_output_spec(
        self, src, src_mask=None, pos_embed=None, output_attentions=False
    ):
        if not self.encoder_layer:
            if output_attentions:
                return src, None
            return src
        encoder_layer_output_spec = self.encoder_layer[0].compute_output_spec(
            hidden_states=src,
            attention_mask=src_mask,
            position_embeddings=pos_embed,
            output_attentions=output_attentions,
        )
        if output_attentions:
            return encoder_layer_output_spec
        return encoder_layer_output_spec

    def call(
        self,
        src,
        src_mask=None,
        pos_embed=None,
        output_attentions=False,
        training=None,
    ):
        current_hidden_tensor = src
        last_layer_attn_weights = None

        for encoder_layer_instance in self.encoder_layer:
            current_hidden_tensor, layer_attn_weights = encoder_layer_instance(
                hidden_states=current_hidden_tensor,
                attention_mask=src_mask,
                position_embeddings=pos_embed,
                output_attentions=output_attentions,
                training=training,
            )
            if output_attentions:
                last_layer_attn_weights = layer_attn_weights

        return current_hidden_tensor, last_layer_attn_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "normalize_before": self.normalize_before,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "num_attention_heads": self.num_attention_heads,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
                "encoder_activation_function": self.encoder_activation_function,
                "activation_dropout": self.activation_dropout_rate,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "num_encoder_layers": self.num_encoder_layers,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
            }
        )
        return config
