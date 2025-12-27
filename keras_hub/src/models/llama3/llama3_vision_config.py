"""Configuration classes for Llama 3.2 Vision model."""

from keras_hub.src.api_export import keras_hub_export

# Default cross-attention layer positions (as per Meta's Llama 3.2 Vision)
DEFAULT_CROSS_ATTENTION_LAYERS = [3, 8, 13, 18, 23, 28, 33, 38]


@keras_hub_export("keras_hub.models.Llama3VisionEncoderConfig")
class Llama3VisionEncoderConfig:
    """Configuration for the Llama 3.2 Vision Encoder.

    This configuration supports both single-stage and two-stage (local+global)
    vision encoder architectures used in Llama 3.2 Vision models.

    Args:
        hidden_dim: int. The size of the transformer hidden state.
        num_layers: int. The number of transformer layers (for single-stage).
            For two-stage, this is the total: local_layers + global_layers.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the feedforward network.
        patch_size: int. The size of each square image patch.
        image_size: int. The maximum width/height of the input images.
        num_channels: int. The number of input channels in the images.
        local_layers: int. Number of local encoder layers (two-stage mode).
        global_layers: int. Number of global encoder layers (two-stage mode).
        activation: str. The activation function to use.
        dropout: float. The dropout value.
        attention_dropout: float. The dropout value for attention.
        layer_norm_epsilon: float. The epsilon value in layer normalization.
    """

    def __init__(
        self,
        hidden_dim=1152,
        num_layers=27,
        num_heads=16,
        intermediate_dim=4304,
        patch_size=14,
        image_size=560,
        num_channels=3,
        local_layers=None,
        global_layers=None,
        activation="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        self._kwargs = kwargs
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.local_layers = local_layers
        self.global_layers = global_layers
        self.activation = activation
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    @property
    def is_two_stage(self):
        """Check if using two-stage encoder architecture."""
        return self.local_layers is not None and self.global_layers is not None

    def get_config(self):
        config = super().get_config() if hasattr(super(), "get_config") else {}
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_channels": self.num_channels,
                "local_layers": self.local_layers,
                "global_layers": self.global_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        config.update(self._kwargs)
        return config


@keras_hub_export("keras_hub.models.Llama3VisionConfig")
class Llama3VisionConfig:
    """Configuration for the Llama 3.2 Vision model.

    This config composes a `Llama3VisionEncoderConfig` and a
    text backbone configuration, plus cross-attention layer positions.

    Args:
        vision_encoder_config: `Llama3VisionEncoderConfig` instance or dict.
        text_config: Text backbone config (dict or config object).
        cross_attention_layers: List of decoder layer indices where
            cross-attention is applied. Default: [3, 8, 13, 18, 23, 28, 33, 38].
        dtype: The dtype for computations.
    """

    def __init__(
        self,
        vision_encoder_config=None,
        text_config=None,
        cross_attention_layers=None,
        dtype=None,
        **kwargs,
    ):
        self._kwargs = kwargs
        super().__init__()

        # Handle Vision Encoder Config
        if vision_encoder_config is None:
            vision_encoder_config = Llama3VisionEncoderConfig()
        elif isinstance(vision_encoder_config, dict):
            vision_encoder_config = Llama3VisionEncoderConfig(
                **vision_encoder_config
            )
        self.vision_encoder_config = vision_encoder_config

        # Handle Text Config
        if text_config is None:
            text_config = {}
        self.text_config = text_config

        # Cross-attention layer positions
        self.cross_attention_layers = (
            cross_attention_layers
            if cross_attention_layers is not None
            else DEFAULT_CROSS_ATTENTION_LAYERS.copy()
        )
        self.dtype = dtype

    def get_config(self):
        config = super().get_config() if hasattr(super(), "get_config") else {}

        # Handle text_config serialization
        text_config_val = self.text_config
        if hasattr(text_config_val, "get_config"):
            text_config_val = text_config_val.get_config()

        config.update(
            {
                "vision_encoder_config": (
                    self.vision_encoder_config.get_config()
                ),
                "text_config": text_config_val,
                "cross_attention_layers": self.cross_attention_layers,
                "dtype": self.dtype,
            }
        )
        config.update(self._kwargs)
        return config


# Preset configurations for Llama 3.2 Vision models
LLAMA_3_2_VISION_11B_CONFIG = {
    "vision_encoder_config": {
        "hidden_dim": 1280,
        "num_layers": 32,
        "num_heads": 16,
        "intermediate_dim": 5120,
        "patch_size": 14,
        "image_size": 560,
        "num_channels": 3,
        "local_layers": 32,
        "global_layers": 8,
    },
    "text_config": {
        "vocabulary_size": 128256,
        "num_layers": 40,
        "hidden_dim": 4096,
        "num_query_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_dim": 14336,
        "rope_max_wavelength": 500000,
        "layer_norm_epsilon": 1e-5,
    },
    "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38],
}

LLAMA_3_2_VISION_90B_CONFIG = {
    "vision_encoder_config": {
        "hidden_dim": 1280,
        "num_layers": 32,
        "num_heads": 16,
        "intermediate_dim": 5120,
        "patch_size": 14,
        "image_size": 560,
        "num_channels": 3,
        "local_layers": 32,
        "global_layers": 8,
    },
    "text_config": {
        "vocabulary_size": 128256,
        "num_layers": 80,
        "hidden_dim": 8192,
        "num_query_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_dim": 28672,
        "rope_max_wavelength": 500000,
        "layer_norm_epsilon": 1e-5,
    },
    "cross_attention_layers": [
        3,
        8,
        13,
        18,
        23,
        28,
        33,
        38,
        43,
        48,
        53,
        58,
        63,
        68,
        73,
        78,
    ],
}
