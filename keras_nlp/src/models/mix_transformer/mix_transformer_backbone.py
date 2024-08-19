# Copyright 2024 The KerasNLP Authors
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
import numpy as np
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_nlp.src.models.mix_transformer.mix_transformer_layers import (
    HierarchicalTransformerEncoder,
)
from keras_nlp.src.models.mix_transformer.mix_transformer_layers import (
    OverlappingPatchingAndEmbedding,
)


@keras_nlp_export("keras_nlp.models.MiTBackbone")
class MiTBackbone(FeaturePyramidBackbone):
    def __init__(
        self,
        depths,
        include_rescaling=True,
        input_image_shape=(224, 224, 3),
        embedding_dims=None,
        **kwargs,
    ):
        """A Backbone implementing the MixTransformer.

        This architecture to be used as a backbone for the SegFormer
        architecture [SegFormer: Simple and Efficient Design for Semantic
        Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
        [Based on the TensorFlow implementation from DeepVision](
            https://github.com/DavidLandup0/deepvision/tree/main/deepvision/models/classification/mix_transformer)

        Args:
            depths: the number of transformer encoders to be used per stage in the
                network.
            include_rescaling: bool, whether to rescale the inputs. If set
                to `True`, inputs will be passed through a `Rescaling(1/255.0)`
                layer. Defaults to `True`.
            input_image_shape: optional shape tuple, defaults to (224, 224, 3).
            embedding_dims: the embedding dims per hierarchical stage, used as
                the levels of the feature pyramid

        Examples:

        Using the class with a `backbone`:

        ```python
        images = np.ones(shape=(1, 96, 96, 3))
        labels = np.zeros(shape=(1, 96, 96, 1))
        backbone = keras_nlp.models.MiTBackbone.from_preset("mit_b0_imagenet")

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
        drop_path_rate = 0.1
        dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]
        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]
        num_stages = 4

        # === Layers ===
        cur = 0
        patch_embedding_layers = []
        transformer_blocks = []
        layer_norms = []

        for i in range(num_stages):
            patch_embed_layer = OverlappingPatchingAndEmbedding(
                project_dim=embedding_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                name=f"patch_and_embed_{i}",
            )
            patch_embedding_layers.append(patch_embed_layer)

            transformer_block = [
                HierarchicalTransformerEncoder(
                    project_dim=embedding_dims[i],
                    num_heads=blockwise_num_heads[i],
                    sr_ratio=blockwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    name=f"hierarchical_encoder_{i}_{k}",
                )
                for k in range(depths[i])
            ]
            transformer_blocks.append(transformer_block)
            cur += depths[i]
            layer_norms.append(keras.layers.LayerNormalization())

        # === Functional Model ===
        image_input = keras.layers.Input(shape=input_image_shape)
        x = image_input

        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255)(x)

        pyramid_outputs = {}
        for i in range(num_stages):
            # Compute new height/width after the `proj`
            # call in `OverlappingPatchingAndEmbedding`
            stride = 4 if i == 0 else 2
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
        self.depths = depths
        self.include_rescaling = include_rescaling
        self.input_image_shape = input_image_shape
        self.embedding_dims = embedding_dims
        self.pyramid_outputs = pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "include_rescaling": self.include_rescaling,
                "embedding_dims": self.embedding_dims,
                "input_image_shape": self.input_image_shape,
            }
        )
        return config
