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

"""Falcon backbone model."""

import copy
import tensorflow as tf
from tensorflow import keras
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.backbone import Backbone
# from keras_nlp.models.falcon.falcon_presets import backbone_presets
from keras_nlp.models.falcon.falcon_decoder import FalconDecoder
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.FalconBackbone")
class FalconBackbone(Backbone):
    """A Falcon decoder network.

    This class implements a Transformer-based decoder model as described in
    ["Falcon: Open Pre-trained Transformer Language Models"]().
    The default constructor gives a fully customizable, randomly initialized Falcon
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://huggingface.co/tiiuae/).
    
    Args:
        vocab_size (int): The vocabulary size.
        hidden_size (int): The hidden size of the model.
        n_layer (int): The number of hidden layers.
        n_head (int): The number of attention heads.
        layer_norm_epsilon (float): The epsilon value for layer normalization.
        initializer_range (float): The range for weight initialization.
        use_cache (bool): Whether to use cache for decoding.
        bos_token_id (int): The ID of the beginning-of-sentence token.
        eos_token_id (int): The ID of the end-of-sentence token.
        apply_residual_connection_post_layernorm (bool): Whether to apply residual connection after layer normalization.
        hidden_dropout (float): The dropout rate for hidden layers.
        attention_dropout (float): The dropout rate for attention layers.
        multi_query (bool): Whether to use multi-query attention.
        alibi (bool): Whether to use Alibi attention.
        bias (bool): Whether to use bias in attention layers.
        parallel_attn (bool): Whether to use parallel attention.

    Examples:
    ```python
    input_data = {
        "token_ids": tf.ones(shape=(1, 12), dtype="int64"),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }

    # Pretrained Falcon model
    model = keras_nlp.models.FalconBackbone.from_preset("falcon_base")
    model(input_data)

    # Randomly initialized Falcon encoder with custom config.
    model = keras_nlp.models.FalconBackbone(
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        multi_query=False,
        alibi=False,
        bias=False,
        parallel_attn=False,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,  # num_hidden_layers
        n_head=8,  # num_attention_heads
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        multi_query=False,
        alibi=False,
        bias=False,
        parallel_attn=False,
        **kwargs,
    ):
        """
        Initialize the FalconBackbone model.

        Args:
            vocab_size (int): The vocabulary size.
            hidden_size (int): The hidden size of the model.
            n_layer (int): The number of hidden layers.
            n_head (int): The number of attention heads.
            layer_norm_epsilon (float): The epsilon value for layer normalization.
            initializer_range (float): The range for weight initialization.
            use_cache (bool): Whether to use cache for decoding.
            bos_token_id (int): The ID of the beginning-of-sentence token.
            eos_token_id (int): The ID of the end-of-sentence token.
            apply_residual_connection_post_layernorm (bool): Whether to apply residual connection after layer normalization.
            hidden_dropout (float): The dropout rate for hidden layers.
            attention_dropout (float): The dropout rate for attention layers.
            multi_query (bool): Whether to use multi-query attention.
            alibi (bool): Whether to use Alibi attention.
            bias (bool): Whether to use bias in attention layers.
            parallel_attn (bool): Whether to use parallel attention.

        Returns:
            None
        """
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        x = token_id_input
        config = self.get_config()
        for _ in range(n_layer):
            x = FalconDecoder(config)(
                hidden_states=x,
                alibi=None,
                attention_mask=None,
                layer_past=None,
                head_mask=None,
                use_cache=use_cache,
                output_attentions=False,
                training=False
            )

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask,
            },
            outputs=x,
            **kwargs,
        )
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.multi_query = multi_query
        self.alibi = alibi
        self.bias = bias
        self.parallel_attn = parallel_attn

    def get_config(self):
        """
        Get the configuration of the FalconBackbone model.

        Returns:
            dict: The model configuration.
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "n_head": self.n_head,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "hidden_dropout": self.hidden_dropout,
                "attention_dropout": self.attention_dropout,
                "apply_residual_connection_post_layernorm": self.apply_residual_connection_post_layernorm,
                "parallel_attn": self.parallel_attn,
            }
        )
        return config

    @property
    def head_dim(self):
        """
        Get the head dimension of the FalconBackbone model.

        Returns:
            int: The head dimension.
        """
        return self.hidden_size // self.n_head

    @property
    def rotary(self):
        """
        Check if the FalconBackbone model uses rotary attention.

        Returns:
            bool: True if rotary attention is used, False otherwise.
        """
        return not self.alibi

    # @classproperty
    # def presets(cls):
    #     """
    #     Get the presets for the FalconBackbone model.

    #     Returns:
    #         dict: The presets for the model.
    #     """
    #     return copy.deepcopy(backbone_presets)
