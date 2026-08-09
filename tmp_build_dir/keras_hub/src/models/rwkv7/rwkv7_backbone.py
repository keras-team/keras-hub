import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.rwkv7.rwkv7_layer import RWKV7_Block


def rwkv7_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.RWKV7Backbone")
class RWKV7Backbone(Backbone):
    """The RWKV7 Transformer core architecture with hyperparameters.

    This network implements a RNN-based decoder network,
    Goose, as described in
    [RWKV-7](https://arxiv.org/abs/2503.14456).

    This network implements a Modern RNN architecture based on linear
    attention mechanisms with recurrent processing, as described in the
    RWKV papers. It includes the embedding lookups and RWKV-7 blocks.

    The default constructor gives a fully customizable, randomly initialized
    RWKV-7 model with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        hidden_size: int. The size of the transformer encoding and pooling
            layers.
        head_size: int. The size of each attention head.
        num_layers: int. The number of transformer layers.
        vocabulary_size: int. The size of the token vocabulary.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        gate_lora: int. LoRA dimension for gating. Defaults to `128` .
        mv_lora: int. LoRA dimension for value mixing. Defaults to `32` .
        aaa_lora: int. LoRA dimension for alpha parameters.Defaults to `64` .
        decay_lora: int. LoRA dimension for decay parameters.Defaults to `64` .
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.
        dropout_rate: float. Dropout rate for the dropout layer.

    Examples:

    ```python
    input_data = np.ones(shape=(1, 12), dtype="int32")


    # Randomly initialized RWKV-7 decoder with custom config.
    model = keras_hub.models.RWKV7Backbone(
        vocabulary_size=10,
        hidden_size=512,
        num_layers=2,
        head_size=64,
        intermediate_dim=1024,
        dtype="float32"
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        hidden_size,
        head_size,
        num_layers,
        vocabulary_size,
        intermediate_dim,
        gate_lora=128,
        mv_lora=32,
        aaa_lora=64,
        decay_lora=64,
        dtype=None,
        dropout_rate=0,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_size,
            embeddings_initializer=rwkv7_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )

        self.output_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="output_norm",
            dtype=dtype,
        )
        self.dropout = keras.layers.Dropout(
            dropout_rate,
            dtype=dtype,
            name="dropout",
        )
        self.rwkv_layers = []
        for i in range(num_layers):
            layer = RWKV7_Block(
                hidden_size,
                head_size,
                intermediate_dim,
                gate_lora,
                mv_lora,
                aaa_lora,
                decay_lora,
                use_initial_norm=i == 0,
                kernel_initializer=rwkv7_kernel_initializer(),
                dtype=dtype,
                name=f"rwkv_layer_{i}",
            )

            self.rwkv_layers.append(layer)
        self.head = keras.layers.Dense(
            units=vocabulary_size,
            kernel_initializer=rwkv7_kernel_initializer(),
            use_bias=False,
            name="head",
            dtype=dtype,
        )
        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )

        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        x = self.token_embedding(token_id_input)
        padding_mask = ops.cast(padding_mask_input, dtype=x.dtype)
        v_first = None
        for rwkv_layer in self.rwkv_layers:
            x, v_first = rwkv_layer(x, v_first, padding_mask)
            x = self.dropout(x)
        sequence_output = self.output_layer_norm(x)
        sequence_output = self.head(sequence_output)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        self.num_layers = num_layers
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        self.vocabulary_size = vocabulary_size
        self.dropout_rate = dropout_rate
        self.intermediate_dim = intermediate_dim

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "head_size": self.head_size,
                "gate_lora": self.gate_lora,
                "mv_lora": self.mv_lora,
                "aaa_lora": self.aaa_lora,
                "decay_lora": self.decay_lora,
                "vocabulary_size": self.vocabulary_size,
                "dropout_rate": self.dropout_rate,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
            }
        )
        return config
