# Copyright 2024 The KerasHub Authors
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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock
from keras_hub.src.models.gemma.rms_normalization import RMSNormalization


@keras_hub_export("keras_hub.models.GemmaBackbone")
class GemmaBackbone(Backbone):
    """Gemma core network with hyperparameters.

    This backbone implements the base Transformer network for the Gemma model.
    It includes the embedding lookups and transformer layers. This backbone
    will output the final hidden states for each token, not generative
    predictions over the vocabulary space. For a higher-level object for text
    generation, see `keras_hub.models.GemmaCausalLM`.

    The default constructor gives a fully customizable, randomly initialized
    Gemma model with any number of layers, heads, and embedding dimensions. To
    load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_query_heads: int. The number of heads for the query projections in
            the attention layer.
        num_key_value_heads: int. The number of heads for the key and value
            projections in the attention layer.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        head_dim: int. The size of each attention head.
        layer_norm_epsilon: float. The epsilon value user for every layer norm
            in the transformer model.
        dropout: float. Dropout probability for the Transformer encoder.
        query_head_dim_normalize: boolean. If `True` normalize the query before
            attention with `head_dim`. If `False`, normalize the query with
            `hidden_dim / num_query_heads`. Defaults to True.
        use_post_ffw_norm: boolean. Whether to normalize after the feedforward
            block. Defaults to False.
        use_post_attention_norm: boolean. Whether to normalize after the attention
            block. Defaults to False.
        attention_logit_soft_cap: None or int. Soft cap for the attention logits.
            Defaults to None.
        final_logit_soft_cap: None or int. Soft cap for the final logits.
            Defaults to None.
        use_sliding_window_attention boolean. Whether to use sliding local
          window attention. Defaults to False.
        sliding_window_size: int. Size of the sliding local window. Defaults to
            4096.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Gemma decoder.
    model = keras_hub.models.GemmaBackbone.from_preset("gemma_2b_en")
    model(input_data)

    # Randomly initialized Gemma decoder with custom config.
    model = keras_hub.models.GemmaBackbone(
        vocabulary_size=50257,
        num_layers=12,
        num_query_heads=12,
        num_key_value_heads=1,
        hidden_dim=768,
        intermediate_dim=3072,
        head_dim=64,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        query_head_dim_normalize=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
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
            logit_soft_cap=final_logit_soft_cap,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            sliding_window = use_sliding_window_attention and (i % 2 == 0)
            layer = GemmaDecoderBlock(
                intermediate_dim=intermediate_dim,
                hidden_dim=hidden_dim,
                num_query_heads=num_query_heads,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
                query_head_dim_normalize=query_head_dim_normalize,
                use_post_ffw_norm=use_post_ffw_norm,
                use_post_attention_norm=use_post_attention_norm,
                logit_soft_cap=attention_logit_soft_cap,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
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
        token_id_input = keras.Input(
            shape=(None,), dtype="float32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="float32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.query_head_dim_normalize = query_head_dim_normalize
        self.use_post_ffw_norm = use_post_ffw_norm
        self.use_post_attention_norm = use_post_attention_norm
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.final_logit_soft_cap = final_logit_soft_cap
        self.sliding_window_size = sliding_window_size
        self.use_sliding_window_attention = use_sliding_window_attention

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "query_head_dim_normalize": self.query_head_dim_normalize,
                "use_post_ffw_norm": self.use_post_ffw_norm,
                "use_post_attention_norm": self.use_post_attention_norm,
                "final_logit_soft_cap": self.final_logit_soft_cap,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "sliding_window_size": self.sliding_window_size,
                "use_sliding_window_attention": self.use_sliding_window_attention,
            }
        )
        return config

    @staticmethod
    def get_layout_map(
        device_mesh,
        model_parallel_dim_name="model",
        data_parallel_dim_name="batch",
    ):
        """Get a `keras.distribution.LayoutMap` for model parallel distribution.

        The returned `LayoutMap` contains the sharding spec for the gemma
        backbone weights, so that you can use it to distribute weights across
        the accelerators.

        Example:
        ```
        # Feel free to change the mesh shape to balance data and model parallel
        mesh = keras.distribution.DeviceMesh(
            shape=(1, 8), axis_names=('batch', 'model'),
            devices=keras.distribution.list_devices())
        layout_map = GemmaBackbone.get_layout_map(
            mesh, model_parallel_dim_name="model")

        distribution = keras.distribution.ModelParallel(
            mesh, layout_map, batch_dim_name='batch')
        with distribution.scope():
           gemma_model = keras_hub.models.GemmaCausalLM.from_preset()
        ```

        Args:
            device_mesh: The `keras.distribution.DeviceMesh` instance for
                distribution.
            model_parallel_dim_name: The axis name of the device mesh, where
                the weights should be partition on.
            data_parallel_dim_name: The axis name of the device mesh, where
                the data should be partition on.
        Return:
            `keras.distribution.LayoutMap` that contains the sharding spec
            of all the model weights.
        """
        # The weight path and shape of the Gemma backbone is like below (for 2G)
        # token_embedding/embeddings,  (256128, 2048), 524550144
        # repeat block for decoder
        # ...
        # decoder_block_17/pre_attention_norm/scale,  (2048,), 2048
        # decoder_block_17/attention/query/kernel,  (8, 2048, 256), 4194304
        # decoder_block_17/attention/key/kernel,  (8, 2048, 256), 4194304
        # decoder_block_17/attention/value/kernel,  (8, 2048, 256), 4194304
        # decoder_block_17/attention/attention_output/kernel,  (8, 256, 2048), 4194304
        # decoder_block_17/pre_ffw_norm/scale,  (2048,), 2048
        # decoder_block_17/ffw_gating/kernel,  (2048, 16384), 33554432
        # decoder_block_17/ffw_gating_2/kernel,  (2048, 16384), 33554432
        # decoder_block_17/ffw_linear/kernel,  (16384, 2048), 33554432
        if not isinstance(device_mesh, keras.distribution.DeviceMesh):
            raise ValueError(
                "Invalid device_mesh type. Expected `keras.distribution.Device`,"
                f" got {type(device_mesh)}"
            )
        if model_parallel_dim_name not in device_mesh.axis_names:
            raise ValueError(
                f"{model_parallel_dim_name} is not found in the "
                f"device_mesh.axis_names. {device_mesh.axis_name=}"
            )
        if data_parallel_dim_name not in device_mesh.axis_names:
            raise ValueError(
                f"{data_parallel_dim_name} is not found in the "
                f"device_mesh.axis_names. {device_mesh.axis_name=}"
            )
        # Note that it is possible to further config the mesh to be 3D, eg
        # (data, seq, model). We leave it as 2D for now for simplicity.
        data_dim = data_parallel_dim_name
        model_dim = model_parallel_dim_name
        # The sharding config is based on the Gemma team training config.
        # See https://arxiv.org/abs/2403.08295
        layout_map = keras.distribution.LayoutMap(device_mesh)
        layout_map["token_embedding/embeddings"] = (model_dim, data_dim)
        layout_map["decoder_block.*attention.*(query|key|value).kernel"] = (
            model_dim,
            data_dim,
            None,
        )
        layout_map["decoder_block.*attention_output.kernel"] = (
            model_dim,
            None,
            data_dim,
        )
        layout_map["decoder_block.*ffw_gating.kernel"] = (data_dim, model_dim)
        layout_map["decoder_block.*ffw_gating_2.kernel"] = (data_dim, model_dim)
        layout_map["decoder_block.*ffw_linear.kernel"] = (model_dim, data_dim)

        return layout_map
