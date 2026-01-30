import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor


@keras_hub_export("keras_hub.models.MoondreamPreprocessor")
class MoondreamPreprocessor(CausalLMPreprocessor):
    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter

    def call(self, x, y=None, sample_weight=None):
        output = super().call(x, y, sample_weight)

        # 1. Identify the input dictionary from the output
        # If output is a tuple (x, y, sw), the first element is the input dict.
        if isinstance(output, tuple):
            x_out = output[0]
        else:
            x_out = output

        # 2. Type Guard for Pylance
        # We explicitly check if x_out IS a dictionary.
        # This stops Pylance from thinking it might be a Tuple/List.
        if isinstance(x_out, dict) and isinstance(x, dict) and "images" in x:
            images = x["images"]
            if self.image_converter:
                images = self.image_converter(images)
            x_out["images"] = images

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": keras.saving.serialize_keras_object(
                    self.image_converter
                ),
            }
        )
        return config
