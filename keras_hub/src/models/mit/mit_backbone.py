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
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.models.mit.mit_layers import HierarchicalTransformerEncoder
from keras_hub.src.models.mit.mit_layers import OverlappingPatchingAndEmbedding


@keras_hub_export("keras_hub.models.MiTBackbone")
class MiTBackbone(FeaturePyramidBackbone):
    def __init__(
        self,
        layerwise_depths,
        num_layers,
        layerwise_num_heads,
        layerwise_sr_ratios,
        max_drop_path_rate,
        layerwise_patch_sizes,
        layerwise_strides,
        image_shape=(None, None, 3),
        hidden_dims=None,
        **kwargs,
    ):
        """A Backbone implementing the MixTransformer.

        This architecture to be used as a backbone for the SegFormer
        architecture [SegFormer: Simple and Efficient Design for Semantic
        Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
        [Based on the TensorFlow implementation from DeepVision](
            https://github.com/DavidLandup0/deepvision/tree/main/deepvision/models/classification/mix_transformer)

        Args:
            layerwise_depths: The number of transformer encoders to be used per
                layer in the network.
            num_layers: int. The number of Transformer layers.
            layerwise_num_heads: list of integers, the number of heads to use
            in the attention computation for each layer.
            layerwise_sr_ratios: list of integers, the sequence reduction
                ratio to perform for each layer on the sequence before key and
                value projections. If set to > 1, a `Conv2D` layer is used to
                reduce the length of the sequence.
            max_drop_path_rate: The final value of the `linspace()` that
                defines the drop path rates for the `DropPath` layers of
                the `HierarchicalTransformerEncoder` layers.
            image_shape: optional shape tuple, defaults to (None, None, 3).
            hidden_dims: the embedding dims per hierarchical layer, used as
                the levels of the feature pyramid.
            patch_sizes: list of integers, the patch_size to apply for each
                layer.
            strides: list of integers, stride to apply for each layer.

        Examples:

        Using the class with a `backbone`:

        ```python
        images = np.ones(shape=(1, 96, 96, 3))
        labels = np.zeros(shape=(1, 96, 96, 1))
        backbone = keras_hub.models.MiTBackbone.from_preset("mit_b0_ade20k_512")

        # Evaluate model
        model(images)

        # Train model
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        model.fit(images, labels, epochs=3)
        ```
        """
        dpr = [
            x
            for x in np.linspace(0.0, max_drop_path_rate, sum(layerwise_depths))
        ]

        # === Layers ===
        cur = 0
        patch_embedding_layers = []
        transformer_blocks = []
        layer_norms = []

        for i in range(num_layers):
            patch_embed_layer = OverlappingPatchingAndEmbedding(
                project_dim=hidden_dims[i],
                patch_size=layerwise_patch_sizes[i],
                stride=layerwise_strides[i],
                name=f"patch_and_embed_{i}",
            )
            patch_embedding_layers.append(patch_embed_layer)

            transformer_block = [
                HierarchicalTransformerEncoder(
                    project_dim=hidden_dims[i],
                    num_heads=layerwise_num_heads[i],
                    sr_ratio=layerwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    name=f"hierarchical_encoder_{i}_{k}",
                )
                for k in range(layerwise_depths[i])
            ]
            transformer_blocks.append(transformer_block)
            cur += layerwise_depths[i]
            layer_norms.append(keras.layers.LayerNormalization(epsilon=1e-5))

        # === Functional Model ===
        image_input = keras.layers.Input(shape=image_shape)
        x = image_input  # Intermediate result.
        pyramid_outputs = {}
        for i in range(num_layers):
            # Compute new height/width after the `proj`
            # call in `OverlappingPatchingAndEmbedding`
            stride = layerwise_strides[i]
            new_height, new_width = (
                int(ops.shape(x)[1] / stride),
                int(ops.shape(x)[2] / stride),
            )

            x = patch_embedding_layers[i](x)
            for blk in transformer_blocks[i]:
                x = blk(x)
            x = layer_norms[i](x)
            x = keras.layers.Reshape(
                (new_height, new_width, -1), name=f"output_level_{i}"
            )(x)
            pyramid_outputs[f"P{i + 1}"] = x

        super().__init__(inputs=image_input, outputs=x, **kwargs)

        # === Config ===
        self.layerwise_depths = layerwise_depths
        self.image_shape = image_shape
        self.hidden_dims = hidden_dims
        self.pyramid_outputs = pyramid_outputs
        self.num_layers = num_layers
        self.layerwise_num_heads = layerwise_num_heads
        self.layerwise_sr_ratios = layerwise_sr_ratios
        self.max_drop_path_rate = max_drop_path_rate
        self.layerwise_patch_sizes = layerwise_patch_sizes
        self.layerwise_strides = layerwise_strides

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layerwise_depths": self.layerwise_depths,
                "hidden_dims": self.hidden_dims,
                "image_shape": self.image_shape,
                "num_layers": self.num_layers,
                "layerwise_num_heads": self.layerwise_num_heads,
                "layerwise_sr_ratios": self.layerwise_sr_ratios,
                "max_drop_path_rate": self.max_drop_path_rate,
                "layerwise_patch_sizes": self.layerwise_patch_sizes,
                "layerwise_strides": self.layerwise_strides,
            }
        )
        return config
