import tensorflow as tf
from tensorflow import keras

class RandomWordDeletion(keras.layers.Layer):
    """Augments input by randomly deleting words

    Args:
        probability: probability of a word being chosen for deletion
        max_deletions: The maximum number of words to replace

    Examples:

    Basic usage.
    >>> augmenter = keras_nlp.layers.RandomWordDeletion(
    ...     probability = 1,
    ...     max_deletions = 1,
    ... )
    >>> augmenter(["dog dog dog dog dog"])
    <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'dog dog dog dog'], dtype=object)>
    """

    def __init__(
        self, 
        probability, 
        max_deletions, 
        **kwargs):
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer and dtype != tf.string:
                raise ValueError(
                    "Output dtype must be an integer type or a string. "
                    f"Received: dtype={dtype}"
                )

        super().__init__(**kwargs)
        self.probability = probability
        self.max_deletions = max_deletions

    def call(self, inputs):
        """Augments input by randomly deleting words.

        Args:
            inputs: A tensor or nested tensor of strings to augment.

        Returns:
            A tensor or nested tensor of augmented strings.
        """

        isString = False
        if isinstance(inputs, str):
          inputs = [inputs]
          isString = True

        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)
            inputs = tf.cast(inputs, tf.string)

        def _map_fn(inputs):
            print
            scalar_input = inputs.shape.rank == 0
            if scalar_input:
                inputs = tf.expand_dims(inputs, 0)

            ragged_words = tf.strings.split(inputs)

            positions_flat = tf.range(tf.size(ragged_words.flat_values))
            positions = ragged_words.with_flat_values(positions_flat)

            # Figure out how many we are going to select.
            word_counts = tf.cast(ragged_words.row_lengths(), "float32")
            num_to_select = tf.random.stateless_binomial(
                shape=tf.shape(word_counts),
                seed=tf.random.get_global_generator().make_seeds()[:, 0],
                counts=word_counts,
                probs=self.probability,
            )
            num_to_select = tf.math.minimum(num_to_select, self.max_deletions)
            num_to_select = tf.cast(num_to_select, "int64")

            # Shuffle and trim to items that are going to be selected.
            def _shuffle_and_trim(x):
                positions, top_n = x
                shuffled = tf.random.shuffle(positions)
                return shuffled[:top_n]

            selected_for_mask = tf.map_fn(
                _shuffle_and_trim, (positions, num_to_select),
                fn_output_signature=tf.RaggedTensorSpec(
                    ragged_rank=positions.ragged_rank - 1, dtype=positions.dtype))
            selected_for_mask.flat_values.set_shape([None])

            # Construct the mask which is a boolean RT
            # Scatter 0's to positions that have been selector for deletion.
            update_values = tf.zeros_like(selected_for_mask.flat_values, "int32")
            update_indices = selected_for_mask.flat_values
            update_indices = tf.expand_dims(update_indices, -1)
            update_indices = tf.cast(update_indices, "int32")
            mask_flat = tf.ones_like(ragged_words.flat_values, dtype="int32")
            mask_flat = tf.tensor_scatter_nd_update(mask_flat, update_indices, update_values)
            mask = tf.cast(ragged_words.with_flat_values(mask_flat), "bool")

            ragged_words = tf.ragged.boolean_mask(ragged_words, mask)
            deleted = tf.strings.reduce_join(ragged_words, axis=-1, separator=" ")
            if scalar_input:
                deleted = tf.squeeze(deleted, 0)
            return deleted
        
        if isinstance(inputs, tf.Tensor):
            if () == inputs.get_shape():
                inputs = tf.reshape( tf.map_fn(_map_fn, tf.reshape( inputs, ( 1, ) ) ), () ) 
            else:
                inputs = tf.map_fn(_map_fn, inputs)
        if isString:
            inputs = inputs[0]
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "probability": self.probability,
                "max_deletions": self.max_deletions,
            }
        )
        return config