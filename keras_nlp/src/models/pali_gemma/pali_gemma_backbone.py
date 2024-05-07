# Copyright 2023 The KerasNLP Authors
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
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.backend import config
from keras_nlp.src.backend import keras
from keras_nlp.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.gemma.rms_normalization import RMSNormalization
from keras_nlp.src.models.pali_gemma.pali_gemma_decoder_block import (
    PaliGemmaDecoderBlock,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_vit import PaliGemmaVit


@keras_nlp_export("keras_nlp.models.PaliGemmaBackbone")
class PaliGemmaBackbone(Backbone):
    def __init__(
        self,
        vocabulary_size=257152,
        image_size=224,
        num_layers=18,
        num_query_heads=8,
        num_key_value_heads=1,
        hidden_dim=2048,
        intermediate_dim=32768,
        head_dim=256,
        layer_norm_epsilon=1e-6,
        dropout=0,
        vit_patch_size=14,
        vit_num_heads=16,
        vit_hidden_dim=1152,
        vit_num_layers=27,
        vit_intermediate_dim=4304,
        vit_pooling=None,
        vit_num_classes=2048,
        vit_classifier_activation=None,
        vit_include_rescaling=False,
        vit_name=None,
        dtype=None,
        **kwargs,
    ):
        if not config.keras_3():
            raise ValueError(
                "`PaliGemmaBackbone` requires Keras 3. Run "
                "`pip install -U keras` to upgrade your Keras version, or see "
                "https://keras.io/getting_started/ "
                "for more info on Keras versions and installation."
            )

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
                seed=None,
            ),
            dtype=dtype,
            name="token_embedding",
        )
        self.vit_encoder = PaliGemmaVit(
            image_size=image_size,
            patch_size=vit_patch_size,
            num_heads=vit_num_heads,
            hidden_dim=vit_hidden_dim,
            num_layers=vit_num_layers,
            intermediate_dim=vit_intermediate_dim,
            pooling=vit_pooling,
            num_classes=hidden_dim,
            classifier_activation=vit_classifier_activation,
            include_rescaling=vit_include_rescaling,
            dtype=dtype,
            name=vit_name,
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = PaliGemmaDecoderBlock(
                img_sequence_length=self.vit_encoder.output_token_length,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
                dropout=dropout,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        # === Functional Model ===
        image_input = self.vit_encoder.inputs[0]
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="float32", name="padding_mask"
        )
        img_embeddings = self.vit_encoder(image_input)
        text_embeddings = self.token_embedding(token_id_input)
        x = keras.ops.concatenate((img_embeddings, text_embeddings), axis=1)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        # VIT Params
        self.vit_patch_size = vit_patch_size
        self.vit_num_heads = vit_num_heads
        self.vit_hidden_dim = vit_hidden_dim
        self.vit_num_layers = vit_num_layers
        self.vit_intermediate_dim = vit_intermediate_dim
        self.vit_pooling = vit_pooling
        self.vit_num_classes = vit_num_classes
        self.vit_classifier_activation = vit_classifier_activation
        self.vit_include_rescaling = vit_include_rescaling
        self.vit_name = vit_name

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "vit_patch_size": self.vit_patch_size,
                "vit_num_heads": self.vit_num_heads,
                "vit_hidden_dim": self.vit_hidden_dim,
                "vit_num_layers": self.vit_num_layers,
                "vit_pooling": self.vit_pooling,
                "vit_num_classes": self.vit_num_classes,
                "vit_classifier_activation": self.vit_classifier_activation,
                "vit_include_rescaling": self.vit_include_rescaling,
                "vit_name": self.vit_name,
            }
        )
        return config
