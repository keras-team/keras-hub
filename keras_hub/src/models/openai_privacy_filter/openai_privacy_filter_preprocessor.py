import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.models.preprocessor import Preprocessor


@keras_hub_export(
    "keras_hub.models.OpenAIPrivacyFilterPreprocessor",
)
class OpenAIPrivacyFilterPreprocessor(Preprocessor):
    """Preprocessor for OpenAI Privacy Filter token classification.

    This preprocessing layer tokenizes and pads/truncates input text into
    `token_ids` and `padding_mask` tensors suitable for the backbone.

    Args:
        tokenizer: A `OpenAIPrivacyFilterTokenizer` instance.
        sequence_length: int. The length of the packed inputs.
            Defaults to `512`.
        truncate: string. The algorithm to truncate a list of batched
            segments. Defaults to `"round_robin"`.
    """

    backbone_cls = OpenAIPrivacyFilterBackbone
    tokenizer_cls = OpenAIPrivacyFilterTokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self._sequence_length = sequence_length
        self.truncate = truncate

        # Since `MultiSegmentPacker` requires TF workflow, we
        # currently disable the Python workflow for
        # `OpenAIPrivacyFilterPreprocessor`.
        self.tokenizer._allow_python_workflow = False

    def build(self, input_shape):
        super().build(input_shape)
        # Use pad_token_id for start/end/sep since this encoder model
        # does not use special start/end tokens.
        pad_id = self.tokenizer.pad_token_id
        self.packer = MultiSegmentPacker(
            start_value=pad_id,
            end_value=pad_id,
            pad_value=pad_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        x = x if isinstance(x, tuple) else (x,)
        x = tuple(self.tokenizer(segment) for segment in x)
        token_ids, segment_ids = self.packer(x, sequence_length=sequence_length)
        x = {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        if self.packer is not None:
            self.packer.sequence_length = value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "truncate": self.truncate,
            }
        )
        return config
