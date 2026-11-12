import keras
from keras import ops

from keras_hub.src.models.qwen3.qwen3_layernorm import Qwen3LayerNorm
from keras_hub.src.models.qwen3_5.qwen3_5_layers import (
    Qwen3_5InterleaveEmbeddings,
)


class Qwen3VLLayerNorm(Qwen3LayerNorm):
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


class Qwen3VLInterleaveEmbeddings(Qwen3_5InterleaveEmbeddings):
    """Scatter vision embeddings into a text embedding sequence.

    Same as `Qwen3_5InterleaveEmbeddings`, but the vision embeddings are cast
    to the dtype of the text embeddings first, since the vision tower may run
    in a different dtype.
    """

    def call(self, image_embeddings, text_embeddings, vision_indices):
        image_embeddings = ops.cast(image_embeddings, text_embeddings.dtype)
        vision_indices = flat_vision_indices(
            vision_indices, ops.shape(text_embeddings)[1]
        )
        return super().call(image_embeddings, text_embeddings, vision_indices)


def flat_vision_indices(vision_indices, sequence_length):
    """Turn per-sample vision indices into indices of the flattened batch.

    Args:
        vision_indices: int Tensor. `(batch, num_vision_tokens)` positions
            within each sample, or `(num_vision_tokens,)` indices that are
            already flat.
        sequence_length: int. The sequence length of the batch.

    Returns:
        An int32 Tensor of shape `(batch * num_vision_tokens,)`.
    """
    vision_indices = ops.cast(vision_indices, "int32")
    if len(vision_indices.shape) == 1:
        return vision_indices
    batch_size = ops.shape(vision_indices)[0]
    offsets = ops.arange(batch_size, dtype="int32") * sequence_length
    return ops.reshape(vision_indices + offsets[:, None], (-1,))


def _flatten_vision_inputs(features, vision_indices, hidden_dim, seq_len):
    # Flatten per-batch (B, N, D) features to rows and offset the per-sample
    # indices into the flattened batch.
    if len(features.shape) == 3:
        features = ops.reshape(features, (-1, hidden_dim))
    vision_indices = flat_vision_indices(vision_indices, seq_len)
    return features, ops.expand_dims(vision_indices, -1)


class Qwen3VLDeepStackInject(keras.layers.Layer):
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
                with positions within each sample, or `(num_vision_tokens,)`
                with indices into the flattened `(batch * seq_len)`
                sequence.

        Returns:
            Tensor of shape `(batch, seq_len, hidden_dim)`.
        """
        shape = ops.shape(hidden)
        deepstack_feature, vision_indices = _flatten_vision_inputs(
            deepstack_feature, vision_indices, self.hidden_dim, shape[1]
        )
        flat_hidden = ops.reshape(hidden, (-1, self.hidden_dim))
        deepstack_feature = ops.cast(deepstack_feature, flat_hidden.dtype)
        # `scatter_update` writes all rows at once, where the torch
        # `scatter` loops over them, which `torch.export` cannot trace.
        update = ops.scatter_update(
            ops.zeros_like(flat_hidden), vision_indices, deepstack_feature
        )
        return ops.reshape(flat_hidden + update, shape)

    def compute_output_spec(self, hidden, deepstack_feature, vision_indices):
        return keras.KerasTensor(shape=hidden.shape, dtype=hidden.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
