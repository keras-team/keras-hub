import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm


@keras_hub_export("keras_hub.models.Gemma4UnifiedAudioEmbedder")
class Gemma4UnifiedAudioEmbedder(keras.layers.Layer):
    """Lightweight encoder-free audio embedder for Gemma4 Unified (12B).

    Unlike the USM conformer-based `Gemma4AudioEncoder` used by the 26B/31B
    models, this embedder projects raw audio waveform frames directly into
    the language model's hidden space via a single linear projection.
    There is no separate conformer encoder.

    Each audio soft-token corresponds to `audio_embed_dim` raw waveform
    samples (640 samples at 16 kHz ≈ 40 ms of audio).

    This embedder:
    1. Applies a parameter-free RMS norm (`Gemma4VNorm`).
    2. Projects the normalised audio features via a dense layer to `hidden_dim`.

    Args:
        hidden_dim: int. Output embedding dimension (must match the text
            backbone's `hidden_dim`).
        audio_embed_dim: int. Dimension of each audio input frame
            (= `audio_samples_per_token`, typically 640).
        layer_norm_epsilon: float. Epsilon for the pre-projection norm.
            Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. Compute dtype.

    Example:
    ```python
    import numpy as np

    embedder = keras_hub.models.Gemma4UnifiedAudioEmbedder(
        hidden_dim=3840,
        audio_embed_dim=640,
    )
    audio_features = np.ones((1, 100, 640), dtype="float32")
    audio_mask = np.ones((1, 100), dtype="bool")
    output = embedder(audio_features, audio_mask)
    # output.shape == (1, 100, 3840)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        audio_embed_dim=640,
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.audio_embed_dim = audio_embed_dim
        self.layer_norm_epsilon = layer_norm_epsilon

        # Linear projection: audio_embed_dim → hidden_dim
        self.embedding_projection = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="embedding_projection",
        )

        # Pre-projection norm (parameter-free RMSNorm / VNorm).
        # Applied BEFORE the projection, matching HF's
        # Gemma4UnifiedMultimodalEmbedder.embedding_pre_projection_norm.
        self.pre_norm = Gemma4VNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="embedding_pre_projection_norm",
        )

        # Store the input feature size so the backbone can build audio inputs.
        self.input_feat_size = audio_embed_dim

    def build(self, input_shape=None):
        # Force sub-layers to build so Keras sees them as built.
        self.embedding_projection.build((None, None, self.audio_embed_dim))
        self.pre_norm.build((None, None, self.audio_embed_dim))
        super().build(input_shape)

    def call(self, audio_features, audio_mask):
        """Project audio features into the language model's hidden space.

        Args:
            audio_features: Tensor of shape `(batch, num_frames,
                audio_embed_dim)`. Raw waveform frames.
            audio_mask: Boolean tensor of shape `(batch, num_frames)`.
                `True` for valid frames, `False` for padding. Currently
                unused in the forward pass but accepted for API
                compatibility with `Gemma4AudioEncoder`.

        Returns:
            Tensor of shape `(batch, num_frames, hidden_dim)`.
        """
        x = self.pre_norm(audio_features)
        x = self.embedding_projection(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "audio_embed_dim": self.audio_embed_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
