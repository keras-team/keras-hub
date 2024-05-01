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
from keras_nlp.src.models.llama.llama_backbone import LlamaBackbone


# LLaMA 3 shares the same architecture as its predecessors
# So, we simply create an alias for API consistency
@keras_nlp_export("keras_nlp.models.Llama3Backbone")
class Llama3Backbone(LlamaBackbone):
    """
    The Llama Transformer core architecture with hyperparameters.

    This network implements a Transformer-based decoder network,
    Llama, as described in
    ["Llama 7B"](https://arxiv.org/pdf/2310.06825.pdf).
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    Llama model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size (int): The size of the token vocabulary.
        num_layers (int): The number of transformer layers.
        num_query_heads (int): The number of query attention heads for
            each transformer.
        hidden_dim (int): The size of the transformer encoding and pooling layers.
        intermediate_dim (int): The output dimension of the first Dense layer in a
            three-layer feedforward network for each transformer.
        num_key_value_heads (int): The number of key and value attention heads for
            each transformer.
        rope_max_wavelength (int, optional): The maximum angular wavelength of the
            sine/cosine curves, for rotary embeddings. Defaults to `10000`.
        rope_scaling_factor (float, optional): The scaling factor for calculation
            of roatary embedding. Defaults to `1.0`.
        layer_norm_epsilon (float, optional): Epsilon for the layer normalization
            layers in the transformer decoder. Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:

    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Llama decoder.
    model = keras_nlp.models.Llama3Backbone.from_preset("llama3_8b_en")
    model(input_data)

    # Randomly initialized Llama decoder with custom config.
    model = keras_nlp.models.Llama3Backbone(
        vocabulary_size=10,
        hidden_dim=512,
        num_layers=2,
        num_query_heads=32,
        num_key_value_heads=8,
        intermediate_dim=1024,
        layer_norm_epsilon=1e-6,
        dtype="float32"
    )
    model(input_data)
    ```
    """

    pass
