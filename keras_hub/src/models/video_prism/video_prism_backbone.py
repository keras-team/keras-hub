import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.video_prism.video_prism_layers import (
    VideoPrismAttenTokenPoolingLayer,
)
from keras_hub.src.models.video_prism.video_prism_layers import (
    VideoPrismEmbedding,
)
from keras_hub.src.models.video_prism.video_prism_layers import (
    VideoPrismEncoder,
)
from keras_hub.src.models.video_prism.video_prism_layers import (
    VideoPrismFactorizedDecoding,
)
from keras_hub.src.models.video_prism.video_prism_layers import (
    VideoPrismFactorizedReshape,
)
from keras_hub.src.models.video_prism.video_prism_layers import (
    VideoPrismPatchingAndEmbedding,
)
from keras_hub.src.models.video_prism.video_prism_layers import (
    VideoPrismTemporalEmbedding,
)
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.VideoPrismBackbone")
class VideoPrismBackbone(Backbone):
    """VideoPrism backbone for video and multimodal understanding.

    This backbone implements the VideoPrism architecture, a powerful video
    understanding model that uses a factorized encoder design. The model can
    operate in two modes:

    1. **Video-only mode** (`num_text_layers=0`): Contains only a video encoder
       that processes videos through spatial and temporal factorized encoding,
       outputting frame-level video features.
    2. **Multimodal mode** (`num_text_layers>0`): Includes both a video encoder
       and a CoCa-style text encoder, producing aligned video and text
       embeddings suitable for contrastive learning and vision-language tasks.

    The video encoder uses a factorized design that separately processes spatial
    and temporal information for efficiency and scalability.

    Args:
        num_frames: int. The number of frames in the input video sequence.
        patch_size: int. The size of each square patch in the input image.
        hidden_dim: int. The dimensionality of the hidden representations
            throughout the model.
        intermediate_dim: int. The dimensionality of the intermediate layer in
            the feedforward MLP blocks.
        num_heads: int. The number of attention heads for each transformer.
        num_spatial_layers: int. Number of transformer layers in the spatial
            encoder that processes within-frame information.
        num_temporal_layers: int. Number of transformer layers in the temporal
            encoder that processes across-frame information.
        num_auxiliary_layers: int. Number of additional transformer layers
            applied after the factorized video encoder. Only used when
            `num_text_layers > 0`.
        vocabulary_size: int. The size of the token vocabulary. Only required
            when `num_text_layers > 0`. Defaults to `0`.
        num_text_layers: int. The number of transformer encoder layers for the
            text encoder. Set to `0` for video-only mode, or a positive value
            for multimodal mode with both video and text encoders.
            Defaults to `0`.
        dropout_rate: float. Dropout probability for the Transformer encoder.
            Defaults to `0.0`.
        attention_dropout: float. Dropout probability applied to the attention
            weights. Defaults to `0.0`.
        attention_logit_soft_cap: None or float. Soft cap for the attention
            logits. Defaults to `None`.
        image_shape: tuple of ints. The shape of each input frame as
            `(height, width, channels)`. For example, `(288, 288, 3)`.
        layer_norm_epsilon: float. The epsilon for the layer normalization.
            Defaults to `1e-6`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights. Note that some operations,
            such as softmax and layer normalization, will always be performed
            in float32 precision regardless of dtype.

    Returns:
        When `num_text_layers=0` (video-only mode):
            A tensor of shape
            `(batch_size, num_frames, num_patches, hidden_dim)` containing the
            video features for each patch in each frame.

        When `num_text_layers>0` (multimodal mode):
            A dictionary with two keys:
            - `"vision_embeddings"`: A tensor of shape
              `(batch_size, hidden_dim)` containing the pooled and normalized
              video embeddings.
            - `"text_embeddings"`: A tensor of shape `(batch_size, hidden_dim)`
              containing the normalized text embeddings from the final token.

    Example:

    ```python
    # Video-only mode
    backbone = keras_hub.models.VideoPrismBackbone.from_preset(
        "videoprism_public_v1_base"
    )
    # (batch_size, frames, H, W, C)
    pixel_values = np.random.rand(2, 16, 288, 288, 3)
    # (batch_size, num_frames, num_patches, hidden_dim)
    features = backbone.predict(pixel_values)

    # Multimodal mode with text encoder
    token_ids = np.ones((2, 64), dtype="int32")  # (batch_size, seq_len)
    padding_mask = np.ones((2, 64), dtype="int32")  # (batch_size, seq_len)
    backbone = keras_hub.models.VideoPrismBackbone.from_preset(
        "videoprism_lvt_public_v1_base"
    )
    inputs = {
        "pixel_values": pixel_values,
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }
    outputs = backbone.predict(inputs)
    outputs["vision_embeddings"]  # (batch_size, hidden_dim)
    outputs["text_embeddings"]  # (batch_size, hidden_dim)
    ```
    """

    def __init__(
        self,
        num_frames,
        patch_size,
        hidden_dim,
        intermediate_dim,
        num_heads,
        num_spatial_layers,
        num_temporal_layers,
        num_auxiliary_layers,
        vocabulary_size=0,
        num_text_layers=0,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        attention_logit_soft_cap=None,
        layer_norm_epsilon=1e-6,
        image_shape=(288, 288, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if len(image_shape) != 3:
            raise ValueError(
                "`image_shape` must be a tuple of three integers: "
                "(height, width, channels) or (channels, height, width). "
                f"Received: image_shape={image_shape}"
            )
        if data_format == "channels_last":
            height, width, channels = image_shape
        else:
            channels, height, width = image_shape
        has_text_encoder = num_text_layers > 0
        num_patches = (height // patch_size) * (width // patch_size)

        # === Layers ===
        # Vision encoder.
        self.spatial_reshape = VideoPrismFactorizedReshape(
            image_shape=image_shape,
            data_format=data_format,
            dtype=dtype,
            name="spatial_reshape",
        )
        self.spatial_embedding = VideoPrismPatchingAndEmbedding(
            image_size=(height, width),
            patch_size=(patch_size, patch_size),
            hidden_dim=hidden_dim,
            num_channels=channels,
            dtype=dtype,
            name="spatial_patching_and_embedding",
        )
        self.spatial_encoder = VideoPrismEncoder(
            num_layers=num_spatial_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            attention_logit_soft_cap=attention_logit_soft_cap,
            dtype=dtype,
            name="spatial_encoder",
        )
        self.temporal_embedding = VideoPrismTemporalEmbedding(
            num_frames=num_frames,
            hidden_dim=hidden_dim,
            dtype=dtype,
            name="temporal_embedding",
        )
        self.temporal_encoder = VideoPrismEncoder(
            num_layers=num_temporal_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            attention_logit_soft_cap=attention_logit_soft_cap,
            dtype=dtype,
            name="temporal_encoder",
        )
        self.factorized_decoding = VideoPrismFactorizedDecoding(
            num_patches=num_patches,
            num_frames=num_frames,
            hidden_dim=hidden_dim,
            dtype=dtype,
            name="factorized_decoding",
        )
        if has_text_encoder:
            self.video_reshape = keras.layers.Reshape(
                (-1, hidden_dim), dtype=dtype, name="video_reshape"
            )
            if num_auxiliary_layers > 0:
                self.auxiliary_encoder = VideoPrismEncoder(
                    num_layers=num_auxiliary_layers,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    layer_norm_epsilon=layer_norm_epsilon,
                    attention_logit_soft_cap=attention_logit_soft_cap,
                    activation="gelu",
                    is_causal=False,
                    use_final_layernorm=False,
                    dtype=dtype,
                    name="auxiliary_encoder",
                )
            self.video_pooler = VideoPrismAttenTokenPoolingLayer(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                query_dim=hidden_dim,
                num_queries=1,
                dropout_rate=dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="video_pooler",
            )

            # Text encoder.
            self.text_embedding = VideoPrismEmbedding(
                vocabulary_size=vocabulary_size,
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="text_embedding",
            )
            self.text_encoder = VideoPrismEncoder(
                num_layers=num_text_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
                attention_logit_soft_cap=attention_logit_soft_cap,
                activation="relu",  # Text encoder uses ReLU
                is_causal=True,
                dtype=dtype,
                name="text_encoder",
            )
            self.text_layer_normalization = keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="text_layer_normalization",
            )

        # === Functional Model ===
        pixel_value_input = keras.layers.Input(
            shape=(num_frames, *image_shape), name="pixel_values"
        )
        inputs = pixel_value_input
        if has_text_encoder:
            token_id_input = keras.layers.Input(
                shape=(None,), dtype="int32", name="token_ids"
            )
            padding_mask_input = keras.layers.Input(
                shape=(None,), dtype="int32", name="padding_mask"
            )
            inputs = {
                "pixel_values": pixel_value_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            }
        vision_embeddings = self.get_vision_embeddings(pixel_value_input)
        outputs = vision_embeddings
        if has_text_encoder:
            text_embeddings = self.get_text_embeddings(
                token_id_input, padding_mask_input
            )
            outputs = {
                "vision_embeddings": vision_embeddings,
                "text_embeddings": text_embeddings,
            }

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self.num_auxiliary_layers = num_auxiliary_layers
        self.vocabulary_size = vocabulary_size
        self.num_text_layers = num_text_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.layer_norm_epsilon = layer_norm_epsilon
        self.image_shape = image_shape

    def get_vision_embeddings(self, pixel_values):
        """Get the embeddings from the vision encoder."""
        x = self.spatial_reshape(pixel_values)
        x = self.spatial_embedding(x)
        x = self.spatial_encoder(x)
        x = self.temporal_embedding(x)
        x = self.temporal_encoder(x)
        vision_embeddings = self.factorized_decoding(x)
        if hasattr(self, "video_pooler"):
            vision_embeddings = self.video_reshape(vision_embeddings)
            if hasattr(self, "auxiliary_encoder"):
                vision_embeddings = self.auxiliary_encoder(vision_embeddings)
            vision_embeddings = self.video_pooler(vision_embeddings)
            vision_embeddings = ops.squeeze(vision_embeddings, axis=1)
        return vision_embeddings

    def get_text_embeddings(self, token_ids, padding_mask):
        """Get the embeddings from the text encoder."""
        text_embeddings, attention_mask = self.text_embedding(
            token_ids, padding_mask
        )
        text_embeddings = self.text_encoder(
            text_embeddings, attention_mask=attention_mask
        )
        text_embeddings = self.text_layer_normalization(text_embeddings)
        text_embeddings = text_embeddings[:, -1, :]  # Take class token
        return text_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_frames": self.num_frames,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "num_spatial_layers": self.num_spatial_layers,
                "num_temporal_layers": self.num_temporal_layers,
                "num_auxiliary_layers": self.num_auxiliary_layers,
                "vocabulary_size": self.vocabulary_size,
                "num_text_layers": self.num_text_layers,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "image_shape": self.image_shape,
            }
        )
        return config
