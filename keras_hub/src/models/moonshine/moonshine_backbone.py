import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.moonshine.moonshine_encoder import MoonshineEncoder


@keras_hub_export("keras_hub.models.MoonshineBackbone")
class MoonshineBackbone(Backbone):
    """
    Moonshine backbone for speech recognition.

    This class implements a Transformer-based encoder backbone as used in the
    Moonshine ASR system. It comprises a stack of MoonshineEncoderBlock layers
    and a final layer normalization step. The rotary embeddings are computed
    from a fixed range of positional indices produced by the MoonshineArange
    layer, and then applied to each encoder block to add position-aware
    information into the input sequence. The encoder processes a sequence of
    input embeddings and outputs an encoded representation, which can then be
    used by subsequent model components (e.g., a decoder or a prediction head).

    The default constructor provides a fully customizable, randomly initialized
    Moonshine encoder with a user-specified number of layers, hidden dimensions,
    and attention heads. To load preset architectures and weights, use the
    `from_preset()` constructor.

    Args:
        num_layers: int, Number of stacked Moonshine encoder blocks in the
        backbone.
        hidden_dim: int, Dimensionality of the input and output embeddings for
        each token in the sequence.
        intermediate_dim: int, Dimensionality of the projection layer within
        each encoder block's feed-forward network.
        num_heads: int, Number of attention heads in each encoder block's
        multi-headed attention layer. Must evenly divide hidden_dim.
        feedforward_expansion_factor: int, optional, Multiplier applied to
        intermediate_dim to determine the total width of the feed-forward
        network. Defaults to 4.
        use_swiglu_activation: bool, optional, When True, uses SwiGLU activation
        in the feed-forward network for improved performance. When False, uses
        standard activation. Defaults to False.

    Examples:

    ```python
    import numpy as np
    from keras_hub.models.moonshine import MoonshineBackbone

    encoder_sequence = np.random.rand(1, 100, 256).astype("float32")
    sequence_length = np.array([100], dtype="int32")

    backbone = MoonshineBackbone(
        num_layers=6,
        hidden_dim=256,
        intermediate_dim=512,
        num_heads=8,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False,
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
        intermediate_dim,
        num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False,
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
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            feedforward_expansion_factor=feedforward_expansion_factor,
            use_swiglu_activation=use_swiglu_activation,
            name="encoder",
        )

        # ==== Functional Model ====
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
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
            }
        )
        return config
