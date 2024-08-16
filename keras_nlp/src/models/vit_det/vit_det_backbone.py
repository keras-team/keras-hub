# Copyright 2024 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.vit_det.vit_layers import AddPositionalEmbedding
from keras_nlp.src.models.vit_det.vit_layers import ViTDetPatchingAndEmbedding
from keras_nlp.src.models.vit_det.vit_layers import WindowedTransformerEncoder


@keras_nlp_export("keras_nlp.models.ViTDetBackbone")
class ViTDetBackbone(Backbone):
    """An implementation of ViT image encoder.

    The ViTDetBackbone uses a windowed transformer encoder and relative
    positional encodings. The code has been adapted from [Segment Anything
    paper](https://arxiv.org/abs/2304.02643), [Segment Anything GitHub](
    https://github.com/facebookresearch/segment-anything) and [Detectron2](
    https://github.com/facebookresearch/detectron2).

    Args:
        hidden_size (int, optional): The latent dimensionality to be projected
            into in the output of each stacked windowed transformer encoder.
            Defaults to `768`.
        num_layers (int, optional): The number of transformer encoder layers to
            stack in the Vision Transformer. Defaults to `12`.
        intermediate_dim (int, optional): The dimensionality of the hidden Dense
            layer in the transformer MLP head. Defaults to `768*4`.
        num_heads (int, optional): the number of heads to use in the
            `MultiHeadAttentionWithRelativePE` layer of each transformer
            encoder. Defaults to `12`.
        global_attention_layer_indices (list, optional): Indexes for blocks using
            global attention. Defaults to `[2, 5, 8, 11]`.
        input_shape (tuple[int], optional): The size of the input image in
            `(H, W, C)` format. Defaults to `(1024, 1024, 3)`.
        include_rescaling (bool, optional): Whether to rescale the inputs. If
            set to `True`, inputs will be passed through a
            `Rescaling(1/255.0)` layer. Defaults to `False`.
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
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        intermediate_dim,
        num_heads,
        global_attention_layer_indices,
        include_rescaling=True,
        input_shape=(1024, 1024, 3),
        input_tensor=None,
        patch_size=16,
        num_output_channels=256,
        use_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        # === Functional model ===
        img_input = keras.layers.Input(shape=input_shape)
        # Check that the input image is well specified.
        if img_input.shape[-3] is None or img_input.shape[-2] is None:
            raise ValueError(
                "Height and width of the image must be specified"
                " in `input_shape`."
            )
        if img_input.shape[-3] != img_input.shape[-2]:
            raise ValueError(
                "Input image must be square i.e. the height must"
                " be equal to the width in the `input_shape`"
                " tuple/tensor."
            )
        img_size = img_input.shape[-3]
        x = img_input
        if include_rescaling:
            # Use common rescaling strategy across keras_cv
            x = keras.layers.Rescaling(1.0 / 255.0)(x)
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
        x = keras.models.Sequential(
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
        )(x)

        super().__init__(inputs=img_input, outputs=x, **kwargs)

        # === Config ===
        self.patch_size = patch_size
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
        self.input_tensor = input_tensor
        self.include_rescaling = include_rescaling

    @property
    def pyramid_level_inputs(self):
        raise NotImplementedError(
            "The `ViTDetBackbone` model doesn't compute"
            " pyramid level features."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "include_rescaling": self.include_rescaling,
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
                "global_attention_layer_indices": self.global_attention_layer_indices,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
