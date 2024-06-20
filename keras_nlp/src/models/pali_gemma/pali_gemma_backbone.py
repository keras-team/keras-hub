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
import keras
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
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
    """PaliGemma core network with hyperparameters.

    This backbone implements the mixed-modality PaliGemma architecture. It
    contains a Visual Transformer network, as well as text token embedding
    layer, followed by a backend-agnostic concatenation operation to
    construct a sequence of representations of mixed type embeddings (visual
    and textual). Then, the concatenated sequence is passed through a series
    of Mixed Modality Decoder Blocks. The returned value from calling this model
    represents probabilistic values for output tokens.

    For a higher-level object for text-generation,
    see `keras_nlp.models.PaliGemmaCausalLM`.

    The default constructor gives a fully customizable, randomly initialized
    PaliGemma model with any number of vit layers, heads, embedding
    dimensions, and equivalent configuration for Paligemma Decoder layers. To
    load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        image_size: int. The resolution of the image in both width and height.
            Note: input images must be square.
        num_layers: int. The number of transformer mixed decoder layers.
        num_query_heads: int. The number of heads for the query projections in
            the mixed decoder attention layer.
        num_key_value_heads: int. The number of heads for the key and value
            projections in the mixed decoder attention layers.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each mixed transformer layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer decoder block.
        head_dim: int. The size of each attention head in the mixed decoder.
        vit_patch_size: int. The size of each square patch in the input image.
        vit_num_heads: int. The number of attention heads for the vision(image)
            transformer encoder.
        vit_hidden_dim: int. The size of the transformer hidden state at the end
            of each vision transformer layer.
        vit_num_layers: int. The number of vision transformer layers.
        vit_intermediate_dim: int. The output dimension of the first Dense layer
            in a two-layer feedforward network for vision transformer.
        vit_pooling: string. The encoded vision embeddings are pooled using the
            specified polling setting. The accepted values are `"map"`, `"gap"`,
            `"0"` or `"none"`. Defaults to `"none"`.
        vit_classifier_activation: activation function. The activation that
            is used for final output classification in the vision transformer.
        vit_name: string. The name used for vision transformer layers.
        include_rescaling: bool. If true, the image input will be rescaled from
            the range `[0, 255]`, to the range `[0, 1]`.
        layer_norm_epsilon: float. The epsilon value user for every layer norm
            in all transformer blocks.
        dropout: float. Dropout probability for the Transformer decoder blocks.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "images": np.random.uniform(size=(1, 224, 224, 3)),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained PaliGemma decoder.
    model = keras_nlp.models.PaliGemmaBackbone.from_preset("pali_gemma_mix_224")
    model(input_data)

    # Randomly initialized PaliGemma decoder with custom config.
    model = keras_nlp.models.PaliGemmaBackbone(
        vocabulary_size=50257,
        images_size=224,
        num_layers=12,
        num_query_heads=12,
        num_key_value_heads=1,
        hidden_dim=768,
        intermediate_dim=3072,
        head_dim=64,
        vit_patch_size=14,
        vit_num_heads=8,
        vit_hidden_dim=768,
        vit_intermediate_dim=3072,
        vit_num_layers=2,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        image_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        vit_patch_size,
        vit_num_heads,
        vit_hidden_dim,
        vit_num_layers,
        vit_intermediate_dim=None,  # TODO remove default
        vit_pooling=None,
        vit_classifier_activation=None,
        vit_name=None,
        include_rescaling=True,
        layer_norm_epsilon=1e-6,
        dropout=0,
        dtype=None,
        **kwargs,
    ):
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
        # TODO Remove this. Work around for previous serialization bug.
        vit_intermediate_dim = vit_intermediate_dim or 4304
        self.vit_encoder = PaliGemmaVit(
            image_size=image_size,
            include_rescaling=include_rescaling,
            patch_size=vit_patch_size,
            num_heads=vit_num_heads,
            hidden_dim=vit_hidden_dim,
            num_layers=vit_num_layers,
            intermediate_dim=vit_intermediate_dim,
            pooling=vit_pooling,
            num_classes=hidden_dim,
            classifier_activation=vit_classifier_activation,
            dtype=dtype,
            name=vit_name,
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = PaliGemmaDecoderBlock(
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
            shape=(None,), dtype="int32", name="padding_mask"
        )
        response_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="response_mask"
        )
        img_embeddings = self.vit_encoder(image_input)
        text_embeddings = self.token_embedding(token_id_input)
        text_embeddings = text_embeddings * ops.cast(
            ops.sqrt(hidden_dim), text_embeddings.dtype
        )
        x = ops.concatenate((img_embeddings, text_embeddings), axis=1)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                response_mask=response_mask_input,
            )
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
                "response_mask": response_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.include_rescaling = include_rescaling
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
        self.vit_classifier_activation = vit_classifier_activation
        self.vit_name = vit_name
        # Keep the image_sequence_length as a backbone property for easy access.
        self.image_sequence_length = self.vit_encoder.image_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "include_rescaling": self.include_rescaling,
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
                "vit_intermediate_dim": self.vit_intermediate_dim,
                "vit_pooling": self.vit_pooling,
                "vit_classifier_activation": self.vit_classifier_activation,
                "vit_name": self.vit_name,
            }
        )
        return config
