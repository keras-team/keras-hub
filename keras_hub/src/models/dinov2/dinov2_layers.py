import keras
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


class Dinov2PatchAndEmbeddings(keras.layers.Layer):
    """Patches the image and embeds the patches.

    Args:
        image_size: (int, int). Size of the input image.
        patch_size: (int, int). Size of each image patch.
        hidden_dim: int. Dimensionality of the patch embeddings.
        num_channels: int. Number of channels in the input image. Defaults to
            `3`.
        use_class_token: bool. Whether to use class token to be part of
            patch embedding. Defaults to `True`.
        data_format: str. `"channels_last"` or `"channels_first"`. Defaults to
            `None` (which uses `"channels_last"`).
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_channels=3,
        data_format=None,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        num_positions = num_patches + 1

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_positions = num_positions
        self.dropout_rate = dropout_rate
        self.data_format = standardize_data_format(data_format)

    def build(self, input_shape):
        self.mask_token = self.add_weight(
            shape=(1, self.hidden_dim),
            initializer="zeros",
            dtype=self.variable_dtype,
            name="mask_token",
        )
        self.class_token = self.add_weight(
            shape=(
                1,
                1,
                self.hidden_dim,
            ),
            initializer="random_normal",
            dtype=self.variable_dtype,
            name="class_token",
        )
        self.patch_embedding = keras.layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            activation=None,
            dtype=self.dtype_policy,
            data_format=self.data_format,
            name="patch_embedding",
        )
        self.patch_embedding.build(input_shape)
        self.position_embedding = keras.layers.Embedding(
            self.num_positions,
            self.hidden_dim,
            dtype=self.dtype_policy,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            name="position_embedding",
        )
        self.position_embedding.build((1, self.num_positions))
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.position_ids = keras.ops.expand_dims(
            keras.ops.arange(self.num_positions), axis=0
        )
        self.built = True

    def interpolate_pos_encoding(self, embeddings, height, width):
        """Interpolates positional embeddings for different image sizes."""
        num_patches = ops.shape(embeddings)[1] - 1
        num_positions = ops.shape(self.position_embedding)[1] - 1

        # If image size is unchanged, return as is
        if num_patches == num_positions and height == width:
            return self.position_embedding

        class_pos_embed = self.position_embedding[:, :1]  # CLS token position
        patch_pos_embed = self.position_embedding[:, 1:]  # Patch positions

        # Compute new patch grid size
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]
        patch_pos_embed = ops.reshape(
            patch_pos_embed,
            (1, int(num_positions**0.5), int(num_positions**0.5), -1),
        )

        # Interpolate the position embeddings
        patch_pos_embed = keras.layers.Resizing(
            new_height, new_width, interpolation="bicubic"
        )(patch_pos_embed)

        patch_pos_embed = ops.reshape(patch_pos_embed, (1, -1, self.hidden_dim))

        return ops.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def call(self, inputs, bool_masked_pos=None):
        patch_embeddings = self.patch_embedding(inputs)
        if self.data_format == "channels_first":
            patch_embeddings = ops.transpose(
                patch_embeddings, axes=(0, 2, 3, 1)
            )
        embeddings_shape = ops.shape(patch_embeddings)
        patch_embeddings = ops.reshape(
            patch_embeddings, [embeddings_shape[0], -1, embeddings_shape[-1]]
        )
        position_embeddings = self.position_embedding(self.position_ids)
        position_embeddings = self.interpolate_pos_encoding(
            position_embeddings, embeddings_shape[1], embeddings_shape[2]
        )

        class_token = ops.tile(self.class_token, (embeddings_shape[0], 1, 1))
        patch_embeddings = ops.concatenate(
            [class_token, patch_embeddings], axis=1
        )
        embeddings = ops.add(patch_embeddings, position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.num_positions,
            self.hidden_dim,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_channels": self.num_channels,
                "num_patches": self.num_patches,
                "num_positions": self.num_positions,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
