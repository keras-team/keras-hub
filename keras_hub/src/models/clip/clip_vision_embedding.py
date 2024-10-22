from keras import layers
from keras import ops


class CLIPVisionEmbedding(layers.Layer):
    def __init__(self, hidden_dim, patch_size, image_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_positions = num_patches + 1

        self.patch_embedding = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
            name="patch_embedding",
        )
        self.position_embedding = layers.Embedding(
            num_patches + 1, hidden_dim, name="position_embedding"
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
            dtype="int32",
            trainable=False,
            name="position_ids",
        )
        self.patch_embedding.build(input_shape)
        self.position_embedding.build(self.position_ids.shape)

    def call(self, inputs, training=None):
        # TODO: Support channels_first
        x = inputs
        batch_size = ops.shape(x)[0]
        patch_embeddings = self.patch_embedding(x, training=training)
        patch_embeddings = ops.reshape(
            patch_embeddings, (batch_size, -1, self.hidden_dim)
        )
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
        # TODO: Support channels_first
        output_shape = [input_shape[0], None, self.hidden_dim]
        if input_shape[1] is not None and input_shape[2] is not None:
            patch_num = input_shape[1] // self.patch_size
            output_shape[1] = patch_num**2 + 1
        return output_shape
