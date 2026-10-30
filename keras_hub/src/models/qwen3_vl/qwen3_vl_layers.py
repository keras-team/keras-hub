import keras
from keras import ops

from keras_hub.src.models.qwen3.qwen3_layernorm import Qwen3LayerNorm
from keras_hub.src.models.qwen3_5.qwen3_5_layers import (
    Qwen3_5InterleaveEmbeddings,
)


class Qwen3_VLLayerNorm(Qwen3LayerNorm):
    """RMS normalization layer for Qwen3-VL.

    Same as `Qwen3LayerNorm`, but the normalized input is cast back to the
    input dtype before it is multiplied by the learned scale, as in HF's
    `Qwen3VLTextRMSNorm`.
    """

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(self.scale, input_dtype) * ops.cast(x, input_dtype)

    def compute_output_spec(self, x):
        return keras.KerasTensor(x.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"head_dim": self.head_dim})
        return config


class Qwen3_VLInterleaveEmbeddings(Qwen3_5InterleaveEmbeddings):
    """Scatter vision embeddings into a text embedding sequence.

    Same as `Qwen3_5InterleaveEmbeddings`, but the vision embeddings are cast
    to the dtype of the text embeddings first, since the vision tower may run
    in a different dtype.
    """

    def call(self, image_embeddings, text_embeddings, vision_indices):
        image_embeddings = ops.cast(image_embeddings, text_embeddings.dtype)
        return super().call(image_embeddings, text_embeddings, vision_indices)


def _flatten_vision_inputs(features, vision_indices, hidden_dim):
    # Accept per-batch (B, N, D) / (B, N) inputs and flatten them to rows;
    # inputs that are already flat pass through.
    if features.ndim == 3:
        features = ops.reshape(features, (-1, hidden_dim))
    if vision_indices.ndim == 2:
        vision_indices = ops.reshape(vision_indices, (-1,))
    vision_indices = ops.expand_dims(ops.cast(vision_indices, "int32"), -1)
    return features, vision_indices


class Qwen3_VLDeepStackInject(keras.layers.Layer):
    """Add a DeepStack vision feature to the hidden states.

    The feature rows are added to the hidden states at the vision token
    positions, as in HF's `_deepstack_process`.

    Args:
        hidden_dim: int. Size of the hidden states and vision features.
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, hidden, deepstack_feature, vision_indices):
        """Add `deepstack_feature` rows to `hidden` at `vision_indices`.

        Args:
            hidden: Tensor of shape `(batch, seq_len, hidden_dim)`.
            deepstack_feature: Tensor of shape `(batch, num_vision_tokens,
                hidden_dim)` or `(num_vision_tokens, hidden_dim)`.
            vision_indices: int Tensor of shape `(batch, num_vision_tokens)`
                or `(num_vision_tokens,)`. Indices into the flattened
                `(batch * seq_len)` sequence.

        Returns:
            Tensor of shape `(batch, seq_len, hidden_dim)`.
        """
        shape = ops.shape(hidden)
        deepstack_feature, vision_indices = _flatten_vision_inputs(
            deepstack_feature, vision_indices, self.hidden_dim
        )
        flat_hidden = ops.reshape(hidden, (-1, self.hidden_dim))
        deepstack_feature = ops.cast(deepstack_feature, flat_hidden.dtype)
        update = ops.scatter(
            vision_indices, deepstack_feature, ops.shape(flat_hidden)
        )
        return ops.reshape(flat_hidden + update, shape)

    def compute_output_spec(self, hidden, deepstack_feature, vision_indices):
        return keras.KerasTensor(shape=hidden.shape, dtype=hidden.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
