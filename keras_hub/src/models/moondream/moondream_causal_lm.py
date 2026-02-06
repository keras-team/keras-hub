from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.moondream.moondream_backbone import MoondreamBackbone
from keras_hub.src.models.moondream.moondream_preprocessor import (
    MoondreamPreprocessor,
)


@keras_hub_export("keras_hub.models.MoondreamCausalLM")
class MoondreamCausalLM(CausalLM):
    """
    An end-to-end Moondream model for causal language modeling.

    This model wraps `MoondreamBackbone` and handles the complete flow from
    raw inputs (images + text) to generated text output. It provides a
    high-level interface for image captioning and visual question answering.

    Args:
        backbone: A `MoondreamBackbone` instance. The backbone model that
            connects the vision encoder and text decoder.
        preprocessor: A `MoondreamPreprocessor` instance. Handles data
            preprocessing (tokenization and image resizing).
        **kwargs: Standard Keras keyword arguments.

    Example:
    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moondream.moondream_backbone import (
        MoondreamBackbone
    )
    from keras_hub.src.models.moondream.moondream_causal_lm import (
        MoondreamCausalLM
    )

    # 1. Setup Mock Backbone
    images = keras.Input(shape=(None, None, 3), name="images")
    token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(
        shape=(None,), dtype="int32", name="padding_mask"
    )

    outputs = keras.layers.Dense(2048)(token_ids)

    backbone = keras.Model(
        inputs={
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask
        },
        outputs=outputs
    )

    # 2. Instantiate CausalLM
    model = MoondreamCausalLM(backbone=backbone)

    # 3. Run Forward Pass
    inputs = {
        "images": np.random.rand(2, 378, 378, 3),
        "token_ids": np.random.randint(0, 100, (2, 10)),
        "padding_mask": np.ones((2, 10))
    }
    outputs = model(inputs)
    ```
    """

    backbone_cls = MoondreamBackbone
    preprocessor_cls = MoondreamPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        outputs = backbone(inputs)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        self.backbone = backbone
        self.preprocessor = preprocessor
