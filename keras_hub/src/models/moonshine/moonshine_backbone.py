import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.moonshine.moonshine_encoder import MoonshineEncoder


@keras_hub_export("keras_hub.models.MoonshineBackbone")
class MoonshineBackbone(Backbone):
    """
    Moonshine backbone for speech recognition.

    This class implements a Transformer-based encoder backbone as used in the
    Moonshine ASR system. It comprises a stack of MoonshineEncoderBlock layers &
    a final layer normalization step. The rotary embeddings are computed from a
    fixed range of positional indices produced by the MoonshineArange layer,
    and then applied to each encoder block to add position-aware information
    into the input sequence. The encoder processes a sequence of input
    embeddings and outputs an encoded representation, which can then be used by
    subsequent model components (e.g., a decoder or a prediction head).

    The default constructor provides a fully customizable, randomly initialized
    Moonshine encoder with a user-specified number of layers, hidden dimensions,
    and attention heads. To load preset architectures and weights, use the
    `from_preset()` constructor.

    Args:
        num_layers: int. The number of Moonshine encoder blocks to stack.
        hidden_dim: int. The dimensionality of the input embeddings.
        inner_dim: int. The inner (feedforward) dimensionality within each
        encoder block.
        num_heads: int. The number of attention heads for each encoder block.
        The hidden dimension must be divisible by the number of attention
        heads.
        ff_mult: int, optional. Multiplicative factor for the feedforward layer
        width. Defaults to 4.
        ff_swiglu: bool, optional. If True, use the SwiGLU activation in the
        feedforward network of each encoder block. Defaults to False.

    Examples:

    ```python
    import numpy as np
    from keras_hub.models.moonshine import MoonshineBackbone

    encoder_sequence = np.random.rand(1, 100, 256).astype("float32")
    sequence_length = np.array([100], dtype="int32")

    backbone = MoonshineBackbone(
        num_layers=6,
        hidden_dim=256,
        inner_dim=512,
        num_heads=8,
        ff_mult=4,
        ff_swiglu=False,
    )
    outputs = backbone({
        "encoder_sequence": encoder_sequence,
        "sequence_length": sequence_length
    })
    print(outputs["encoder_output"].shape)
    ```
    """

    def __init__(
        self,
        num_layers,
        hidden_dim,
        inner_dim,
        num_heads,
        ff_mult=4,
        ff_swiglu=False,
        **kwargs,
    ):
        encoder_sequence_input = keras.Input(
            shape=[None, hidden_dim], name="encoder_sequence"
        )
        sequence_length_input = keras.Input(
            shape=[], dtype="int32", name="sequence_length"
        )

        # ==== Layers ====
        self.encoder = MoonshineEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            inner_dim=inner_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            ff_swiglu=ff_swiglu,
            name="encoder",
        )

        encoder_output = self.encoder(
            [encoder_sequence_input, sequence_length_input]
        )
        outputs = {"encoder_output": encoder_output}

        super().__init__(
            inputs={
                "encoder_sequence": encoder_sequence_input,
                "sequence_length": sequence_length_input,
            },
            outputs=outputs,
            **kwargs,
        )

        # ==== Config ====
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "inner_dim": self.inner_dim,
                "num_heads": self.num_heads,
                "ff_mult": self.ff_mult,
                "ff_swiglu": self.ff_swiglu,
            }
        )
        return config
