import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.vit_det.vit_layers import AddPositionalEmbedding
from keras_hub.src.models.vit_det.vit_layers import ViTDetPatchingAndEmbedding
from keras_hub.src.models.vit_det.vit_layers import WindowedTransformerEncoder


@keras_hub_export("keras_hub.models.ViTDetBackbone")
class ViTDetBackbone(Backbone):
    """An implementation of ViT image encoder.

    The ViTDetBackbone uses a windowed transformer encoder and relative
    positional encodings. The code has been adapted from [Segment Anything
    paper](https://arxiv.org/abs/2304.02643), [Segment Anything GitHub](
    https://github.com/facebookresearch/segment-anything) and [Detectron2](
    https://github.com/facebookresearch/detectron2).

    Args:
        hidden_size (int): The latent dimensionality to be projected
            into in the output of each stacked windowed transformer encoder.
        num_layers (int): The number of transformer encoder layers to
            stack in the Vision Transformer.
        intermediate_dim (int): The dimensionality of the hidden Dense
            layer in the transformer MLP head.
        num_heads (int): the number of heads to use in the
            `MultiHeadAttentionWithRelativePE` layer of each transformer
            encoder.
        global_attention_layer_indices (list): Indexes for blocks using
            global attention.
        image_shape (tuple[int], optional): The size of the input image in
            `(H, W, C)` format. Defaults to `(None, None, 3)`.
        patch_size (int, optional): the patch size to be supplied to the
            Patching layer to turn input images into a flattened sequence of
            patches. Defaults to `16`.
        num_output_channels (int, optional): The number of channels (features)
            in the output (image encodings). Defaults to `256`.
        use_bias (bool, optional): Whether to use bias to project the keys,
            queries, and values in the attention layer. Defaults to `True`.
        use_abs_pos (bool, optional): Whether to add absolute positional
            embeddings to the output patches. Defaults to `True`.
        use_rel_pos (bool, optional): Whether to use relative positional
            emcodings in the attention layer. Defaults to `True`.
        window_size (int, optional): The size of the window for windowed
            attention in the transformer encoder blocks. Defaults to `14`.
        layer_norm_epsilon (int, optional): The epsilon to use in the layer
            normalization blocks in transformer encoder. Defaults to `1e-6`.

    Examples:
    ```python
    input_data = np.ones((2, 224, 224, 3), dtype="float32")

    # Pretrained ViTDetBackbone backbone.
    model = keras_hub.models.ViTDetBackbone.from_preset("vit_det")
    model(input_data)

    # Randomly initialized ViTDetBackbone backbone with a custom config.
    model = keras_hub.models.ViTDetBackbone(
            image_shape = (16, 16, 3),
            patch_size = 2,
            hidden_size = 4,
            num_layers = 2,
            global_attention_layer_indices = [2, 5, 8, 11],
            intermediate_dim = 4 * 4,
            num_heads = 2,
            num_output_channels = 2,
            window_size = 2,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        intermediate_dim,
        num_heads,
        global_attention_layer_indices,
        image_shape=(None, None, 3),
        patch_size=16,
        num_output_channels=256,
        use_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        # === Functional model ===
        img_input = keras.layers.Input(shape=image_shape, name="images")
        # Check that the input image is well specified.
        if img_input.shape[-3] is None or img_input.shape[-2] is None:
            raise ValueError(
                "Height and width of the image must be specified"
                " in `image_shape`."
            )
        if img_input.shape[-3] != img_input.shape[-2]:
            raise ValueError(
                "Input image must be square i.e. the height must"
                " be equal to the width in the `image_shape`"
                " tuple/tensor."
            )
        img_size = img_input.shape[-3]
        x = img_input
        # VITDet scales inputs based on the standard ImageNet mean/stddev.
        x = (x - ops.array([0.485, 0.456, 0.406], dtype=x.dtype)) / (
            ops.array([0.229, 0.224, 0.225], dtype=x.dtype)
        )
        x = ViTDetPatchingAndEmbedding(
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            embed_dim=hidden_size,
        )(x)
        if use_abs_pos:
            x = AddPositionalEmbedding(img_size, patch_size, hidden_size)(x)
        for i in range(num_layers):
            x = WindowedTransformerEncoder(
                project_dim=hidden_size,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                use_rel_pos=use_rel_pos,
                window_size=(
                    window_size
                    if i not in global_attention_layer_indices
                    else 0
                ),
                input_size=(img_size // patch_size, img_size // patch_size),
            )(x)
        self.neck = keras.models.Sequential(
            [
                keras.layers.Conv2D(
                    filters=num_output_channels, kernel_size=1, use_bias=False
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
                keras.layers.Conv2D(
                    filters=num_output_channels,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
            ]
        )
        x = self.neck(x)

        super().__init__(inputs=img_input, outputs=x, **kwargs)

        # === Config ===
        self.patch_size = patch_size
        self.image_shape = image_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_output_channels = num_output_channels
        self.use_bias = use_bias
        self.use_rel_pos = use_rel_pos
        self.use_abs_pos = use_abs_pos
        self.window_size = window_size
        self.global_attention_layer_indices = global_attention_layer_indices
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "patch_size": self.patch_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "num_output_channels": self.num_output_channels,
                "use_bias": self.use_bias,
                "use_abs_pos": self.use_abs_pos,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "global_attention_layer_indices": (
                    self.global_attention_layer_indices
                ),
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
