from keras import ops


def window_partition(x, height, width, window_size, hidden_dim):
    x = ops.reshape(
        x,
        (
            -1,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            hidden_dim,
        ),
    )
    x = ops.transpose(x, axes=(0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, window_size, window_size, hidden_dim))
    return x


def window_unpartition(x, height, width, window_size, hidden_dim):
    x = ops.reshape(
        x,
        (
            -1,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            hidden_dim,
        ),
    )
    x = ops.transpose(x, axes=(0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, height, width, hidden_dim))
    return x


def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = ops.unstack(boxes, num=4, axis=-1)
    return ops.stack(
        [
            ops.subtract(x_c, ops.multiply(0.5, w)),
            ops.subtract(y_c, ops.multiply(0.5, h)),
            ops.add(x_c, ops.multiply(0.5, w)),
            ops.add(y_c, ops.multiply(0.5, h)),
        ],
        axis=-1,
    )


def concatenate_padded_sequences(
    sequence1, mask1, sequence_len1, sequence2, mask2, sequence_len2, hidden_dim
):
    """Concatenate two sequences with padding masks.

    Args:
        sequence1: A tensor of shape (batch_size, sequence_length1, hidden_dim).
        mask1: A boolean tensor of shape (batch_size, sequence_length1).
        sequence2: A tensor of shape (batch_size, sequence_length2, hidden_dim).
        mask2: A boolean tensor of shape (batch_size, sequence_length2).
        hidden_dim: An integer representing the hidden dimension.

    Returns:
        concatenated_sequence: A tensor of shape
            (batch_size, sequence_length1 + sequence_length2, hidden_dim).
        concatenated_mask: A boolean tensor of shape
            (batch_size, sequence_length1 + sequence_length2).
    """
    batch_size = ops.shape(sequence1)[0]
    max_length = sequence_len1 + sequence_len2

    actual_sequence_1_lengths = ops.sum(ops.cast(mask1, dtype="int32"), axis=1)
    actual_sequence_2_lengths = ops.sum(ops.cast(mask2, dtype="int32"), axis=1)
    final_lengths = ops.add(
        actual_sequence_1_lengths, actual_sequence_2_lengths
    )

    concatenated_mask = ops.less(
        ops.tile(
            ops.expand_dims(ops.arange(max_length, dtype="int32"), axis=0),
            [batch_size, 1],
        ),
        ops.expand_dims(final_lengths, axis=1),
    )
    concatenated_sequence = ops.concatenate(
        [
            sequence1,
            ops.zeros(
                (batch_size, sequence_len2, hidden_dim), dtype=sequence1.dtype
            ),
        ],
        axis=1,
    )

    # Create the indices.
    indices = ops.tile(
        ops.expand_dims(ops.arange(sequence_len2, dtype="int32"), axis=0),
        [batch_size, 1],
    )
    indices = ops.add(
        indices, ops.expand_dims(actual_sequence_1_lengths, axis=-1)
    )
    # Adjust the indices to account for batch dimension.
    to_add = ops.multiply(ops.arange(batch_size, dtype="int32"), max_length)
    indices = ops.add(
        indices, ops.cast(ops.expand_dims(to_add, axis=-1), "int32")
    )
    # `ops.scatter_update` requires 2D indices. We flatten the inputs before
    # scattering and reshape back after.
    flat_concatenated_sequence = ops.scatter_update(
        ops.reshape(concatenated_sequence, (-1, hidden_dim)),
        ops.reshape(indices, (-1, 1)),
        ops.reshape(
            ops.cast(sequence2, concatenated_sequence.dtype), (-1, hidden_dim)
        ),
    )

    concatenated_sequence = ops.reshape(
        flat_concatenated_sequence, (batch_size, -1, hidden_dim)
    )
    return concatenated_sequence, concatenated_mask


def create_bidirectional_mask(input_embeds, attention_mask):
    seq_len = ops.shape(input_embeds)[1]
    attention_mask = attention_mask[:, None, None, :]
    return ops.tile(attention_mask, (1, 1, seq_len, 1))


def inverse_sigmoid(x, eps=1e-3):
    x = ops.clip(x, 0.0, 1.0)
    x1 = ops.maximum(x, eps)
    x2 = ops.maximum(ops.subtract(1.0, x), eps)
    return ops.log(ops.divide(x1, x2))
