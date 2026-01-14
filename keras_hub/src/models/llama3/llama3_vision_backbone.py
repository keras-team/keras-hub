import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_vision_cross_attention import (
    Llama3VisionCrossAttention,
)
from keras_hub.src.models.llama3.llama3_vision_encoder import (
    Llama3VisionEncoder,
)
from keras_hub.src.models.llama3.llama3_vision_projector import (
    Llama3VisionProjector,
)


@keras_hub_export("keras_hub.models.Llama3VisionBackbone")
class Llama3VisionBackbone(Backbone):
    """Llama 3.2 Vision multimodal model with cross-attention fusion.

    This backbone implements the Llama 3.2 Vision architecture which combines
    a vision encoder (based on SigLIP) with the Llama 3 text model using
    cross-attention layers for multimodal fusion. Visual features are injected
    at specific decoder layers through gated cross-attention.

    The default constructor gives a fully customizable, randomly initialized
    Llama 3.2 Vision model with any configuration. To load preset architectures
    and weights, use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers in the text backbone.
        hidden_dim: int. The size of the transformer hidden state in the text
            backbone.
        num_query_heads: int. The number of attention heads for query
            projections in the text backbone.
        num_key_value_heads: int. The number of attention heads for key and
            value projections in the text backbone.
        intermediate_dim: int. The output dimension of the feedforward network
            in the text backbone.
        vision_hidden_dim: int. The size of the vision encoder hidden state.
        vision_num_layers: int. The number of vision transformer layers.
        vision_num_heads: int. The number of attention heads in the vision
            encoder.
        vision_intermediate_dim: int. The output dimension of the feedforward
            network in the vision encoder.
        vision_patch_size: int. The size of each square image patch.
        vision_image_size: int. The input image resolution (height and width).
        vision_num_channels: int. The number of image input channels.
            Defaults to `3`.
        vision_local_layers: int. Number of local encoder layers for two-stage
            architecture. Defaults to `None` (single-stage).
        vision_global_layers: int. Number of global encoder layers for
            two-stage architecture. Defaults to `None` (single-stage).
        vision_output_dim: int. The output dimension of the vision encoder
            after processing (input to projector). Defaults to `None`, which
            uses `vision_hidden_dim`.
        cross_attention_layers: list of int. Layer indices where cross-attention
            is applied. Defaults to `[3, 8, 13, 18, 23, 28, 33, 38]`.
        rope_max_wavelength: int. The maximum angular wavelength of the
            sine/cosine curves used for RoPE. Defaults to `500000`.
        layer_norm_epsilon: float. Epsilon for layer normalization.
            Defaults to `1e-5`.
        dropout: float. Dropout probability. Defaults to `0.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "pixel_values": np.random.uniform(size=(1, 560, 560, 3)),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Llama 3.2 Vision.
    model = keras_hub.models.Llama3VisionBackbone.from_preset(
        "llama3_2_vision_11b"
    )
    model(input_data)

    # Randomly initialized Llama 3.2 Vision with custom config.
    model = keras_hub.models.Llama3VisionBackbone(
        vocabulary_size=128256,
        num_layers=32,
        hidden_dim=4096,
        num_query_heads=32,
        num_key_value_heads=8,
        intermediate_dim=14336,
        vision_hidden_dim=1280,
        vision_num_layers=32,
        vision_num_heads=16,
        vision_intermediate_dim=5120,
        vision_patch_size=14,
        vision_image_size=560,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        hidden_dim,
        num_query_heads,
        num_key_value_heads,
        intermediate_dim,
        vision_hidden_dim,
        vision_num_layers,
        vision_num_heads,
        vision_intermediate_dim,
        vision_patch_size,
        vision_image_size,
        vision_num_channels=3,
        vision_local_layers=None,
        vision_global_layers=None,
        vision_output_dim=None,
        cross_attention_layers=None,
        rope_max_wavelength=500000,
        layer_norm_epsilon=1e-5,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]
        if vision_output_dim is None:
            vision_output_dim = vision_hidden_dim

        # === Layers ===
        self._vision_encoder = Llama3VisionEncoder(
            hidden_dim=vision_hidden_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            intermediate_dim=vision_intermediate_dim,
            patch_size=vision_patch_size,
            image_size=vision_image_size,
            num_channels=vision_num_channels,
            global_layers=vision_global_layers if vision_global_layers else 8,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            dtype=dtype,
            name="vision_encoder",
        )
        self._vision_projector = Llama3VisionProjector(
            input_dim=vision_output_dim,
            output_dim=hidden_dim,
            dtype=dtype,
            name="vision_projector",
        )
        self._text_backbone = Llama3Backbone(
            vocabulary_size=vocabulary_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_dim=intermediate_dim,
            rope_max_wavelength=rope_max_wavelength,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            dtype=dtype,
            name="text_backbone",
        )
        self._cross_attention_blocks = {}
        for layer_idx in cross_attention_layers:
            if layer_idx < num_layers:
                self._cross_attention_blocks[layer_idx] = (
                    Llama3VisionCrossAttention(
                        hidden_dim=hidden_dim,
                        num_heads=num_query_heads,
                        num_key_value_heads=num_key_value_heads,
                        layer_norm_epsilon=layer_norm_epsilon,
                        dtype=dtype,
                        name=f"cross_attention_{layer_idx}",
                    )
                )

        # === Functional Model ===
        image_input = keras.Input(
            shape=(vision_image_size, vision_image_size, vision_num_channels),
            name="pixel_values",
        )
        token_id_input = keras.Input(
            shape=(None,),
            name="token_ids",
            dtype="int32",
        )
        padding_mask_input = keras.Input(
            shape=(None,),
            name="padding_mask",
            dtype="int32",
        )

        vision_features = self._vision_encoder(image_input)
        vision_features = self._vision_projector(vision_features)

        def create_vision_mask(vision_feats):
            batch_size = ops.shape(vision_feats)[0]
            num_patches = ops.shape(vision_feats)[1]
            return ops.ones((batch_size, num_patches), dtype="bool")

        vision_mask = layers.Lambda(
            create_vision_mask, name="create_vision_mask"
        )(vision_features)

        x = self._text_backbone.token_embedding(token_id_input)
        for i, transformer_layer in enumerate(
            self._text_backbone.transformer_layers
        ):
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
            if i in self._cross_attention_blocks:
                x = self._cross_attention_blocks[i](
                    hidden_states=x,
                    vision_features=vision_features,
                    vision_mask=vision_mask,
                )
        x = self._text_backbone.layer_norm(x)

        super().__init__(
            inputs={
                "pixel_values": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_dim = intermediate_dim
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_intermediate_dim = vision_intermediate_dim
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        self.vision_num_channels = vision_num_channels
        self.vision_local_layers = vision_local_layers
        self.vision_global_layers = vision_global_layers
        self.vision_output_dim = vision_output_dim
        self.cross_attention_layers = cross_attention_layers
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

    @property
    def vision_encoder(self):
        return self._vision_encoder

    @property
    def vision_projector(self):
        return self._vision_projector

    @property
    def text_backbone(self):
        return self._text_backbone

    @property
    def cross_attention_blocks(self):
        return self._cross_attention_blocks

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "intermediate_dim": self.intermediate_dim,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_intermediate_dim": self.vision_intermediate_dim,
                "vision_patch_size": self.vision_patch_size,
                "vision_image_size": self.vision_image_size,
                "vision_num_channels": self.vision_num_channels,
                "vision_local_layers": self.vision_local_layers,
                "vision_global_layers": self.vision_global_layers,
                "vision_output_dim": self.vision_output_dim,
                "cross_attention_layers": self.cross_attention_layers,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config

    def freeze_vision_encoder(self):
        """Freeze the vision encoder and projector."""
        self._vision_encoder.trainable = False
        self._vision_projector.trainable = False

    def freeze_text_backbone(self):
        """Freeze the text backbone."""
        self._text_backbone.trainable = False

    def freeze_cross_attention(self):
        """Freeze all cross-attention layers."""
        for ca_block in self._cross_attention_blocks.values():
            ca_block.trainable = False

    def unfreeze_all(self):
        """Unfreeze all model weights."""
        self._vision_encoder.trainable = True
        self._vision_projector.trainable = True
        self._text_backbone.trainable = True
        for ca_block in self._cross_attention_blocks.values():
            ca_block.trainable = True
        if hasattr(self._vision_encoder, "unfreeze_all"):
            self._vision_encoder.unfreeze_all()
