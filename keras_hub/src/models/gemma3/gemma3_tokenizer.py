from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer

# from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone


@keras_hub_export(
    [
        "keras_hub.tokenizers.Gemma3Tokenizer",
        "keras_hub.models.Gemma3Tokenizer",
    ]
)
class Gemma3Tokenizer(GemmaTokenizer):
    # backbone_cls = Gemma3Backbone

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: might have to add some special tokens here.
        self._add_special_token("<start_of_image>", "start_of_image")
        self._add_special_token("<img>", "image_placeholder")
        self._add_special_token("<end_of_image>", "end_of_image")
