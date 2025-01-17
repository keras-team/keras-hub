from keras import layers
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


class CLIPVisionEmbedding(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        patch_size,
        image_size,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        data_format = standardize_data_format(data_format)
        self.data_format = data_format
        num_patches = (image_size // patch_size) ** 2
        self.num_positions = num_patches + 1

        self.patch_embedding = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=data_format,
            use_bias=False,
            dtype=dtype,
            name="patch_embedding",
        )
        self.position_embedding = layers.Embedding(
            num_patches + 1, hidden_dim, dtype=dtype, name="position_embedding"
        )

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            shape=(self.hidden_dim,),
            initializer="random_normal",
            dtype=self.variable_dtype,
            name="class_embedding",
        )
        self.position_ids = self.add_weight(
            shape=(1, self.num_positions),
            initializer="zeros",
            # Let the backend determine the int dtype. For example, tf
            # requires int64 for correct device placement, whereas jax and torch
            # don't.
            dtype=int,
            trainable=False,
            name="position_ids",
        )
        self.patch_embedding.build(input_shape)
        self.position_embedding.build(self.position_ids.shape)

    def call(self, inputs, training=None):
        x = inputs
        batch_size = ops.shape(x)[0]
        patch_embeddings = self.patch_embedding(x, training=training)
        if self.data_format == "channels_last":
            patch_embeddings = ops.reshape(
                patch_embeddings, (batch_size, -1, self.hidden_dim)
            )
        else:
            patch_embeddings = ops.reshape(
                patch_embeddings, (batch_size, self.hidden_dim, -1)
            )
            patch_embeddings = ops.transpose(patch_embeddings, (0, 2, 1))
        class_embeddings = ops.expand_dims(self.class_embedding, axis=(0, 1))
        class_embeddings = ops.tile(class_embeddings, (batch_size, 1, 1))
        position_embeddings = self.position_embedding(self.position_ids)
        embeddings = ops.concatenate(
            [class_embeddings, patch_embeddings], axis=1
        )
        return ops.add(embeddings, position_embeddings)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                output_shape[1] = patch_num**2 + 1
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                output_shape[1] = patch_num**2 + 1
        return output_shape
