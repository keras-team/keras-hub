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
    `Qwen3VLTextRMSNorm`. The scale has the size of the last input axis.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(self.scale, input_dtype) * ops.cast(x, input_dtype)

    def compute_output_spec(self, x):
        return keras.KerasTensor(x.shape, dtype=self.compute_dtype)


class Qwen3VLInterleaveEmbeddings(Qwen3_5InterleaveEmbeddings):
    """Scatter vision embeddings into a text embedding sequence.

    Same as `Qwen3_5InterleaveEmbeddings`, but `vision_indices` may be
    positions within each sample, of shape `(batch, num_vision_tokens)`.
    They are offset by `sample * sequence_length` into indices of the
    flattened batch before the scatter. 1D indices are taken as already
    flat.
    """

    def build(
        self,
        image_embeddings_shape,
        text_embeddings_shape,
        vision_indices_shape,
    ):
        self.built = True

    def call(self, image_embeddings, text_embeddings, vision_indices):
        vision_indices = flat_vision_indices(
            vision_indices, ops.shape(text_embeddings)[1]
        )
        return super().call(image_embeddings, text_embeddings, vision_indices)

    def compute_output_spec(
        self, image_embeddings, text_embeddings, vision_indices
    ):
        return keras.KerasTensor(
            shape=text_embeddings.shape, dtype=self.compute_dtype
        )


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


def add_vision_rows(flat_hidden, features, vision_indices, sequence_length):
    """Add vision feature rows to flattened hidden states.

    Args:
        flat_hidden: Tensor `(batch * sequence_length, hidden_dim)`.
        features: Tensor `(batch, num_vision_tokens, hidden_dim)` or
            `(num_vision_tokens, hidden_dim)`.
        vision_indices: int Tensor, as in `flat_vision_indices`.
        sequence_length: int. The sequence length of the batch.

    Returns:
        `flat_hidden` with each feature row added at its position.
    """
    hidden_dim = ops.shape(flat_hidden)[-1]
    features = ops.reshape(features, (-1, hidden_dim))
    vision_indices = flat_vision_indices(vision_indices, sequence_length)
    # `scatter_update` writes all rows at once, where the torch `scatter`
    # loops over them, which `torch.export` cannot trace. Its `"add"`
    # reduction lowers to an accumulating scatter that LiteRT cannot run.
    # Only padded rows share a position, and those positions are padding,
    # so writing the rows once and adding is enough.
    update = ops.scatter_update(
        ops.zeros_like(flat_hidden),
        ops.expand_dims(vision_indices, -1),
        ops.cast(features, flat_hidden.dtype),
    )
    return flat_hidden + update


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

    def build(
        self, hidden_shape, deepstack_feature_shape, vision_indices_shape
    ):
        self.built = True

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
        flat_hidden = ops.reshape(hidden, (-1, self.hidden_dim))
        output = add_vision_rows(
            flat_hidden, deepstack_feature, vision_indices, shape[1]
        )
        return ops.reshape(output, shape)

    def compute_output_spec(self, hidden, deepstack_feature, vision_indices):
        return keras.KerasTensor(shape=hidden.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
