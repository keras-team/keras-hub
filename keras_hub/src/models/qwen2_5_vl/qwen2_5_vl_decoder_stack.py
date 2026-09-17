import keras

from qwen2_5_vl_decoder_block import Qwen2_5_VLDecoderBlock
from qwen2_5_vl_rms_norm import Qwen2_5_VLRMSNorm


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLDecoderStack(keras.layers.Layer):
    """
    Stack of Qwen2.5-VL decoder blocks followed by a final RMSNorm.
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
        intermediate_size,
        rms_epsilon=1e-6,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.rms_epsilon = rms_epsilon
        self.dropout = dropout

        self.layers_list = [
            Qwen2_5_VLDecoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                rms_epsilon=rms_epsilon,
                dropout=dropout,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]

        self.final_norm = Qwen2_5_VLRMSNorm(
            hidden_size=hidden_size,
            epsilon=rms_epsilon,
            name="final_norm",
        )

    def call(self, hidden_states, attention_mask=None, training=False):
        for layer in self.layers_list:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
            )
        return self.final_norm(hidden_states)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "intermediate_size": self.intermediate_size,
                "rms_epsilon": self.rms_epsilon,
                "dropout": self.dropout,
            }
        )
        return config