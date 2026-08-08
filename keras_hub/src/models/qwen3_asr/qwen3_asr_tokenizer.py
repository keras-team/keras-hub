from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen3ASRTokenizer",
        "keras_hub.models.Qwen3ASRTokenizer",
    ]
)
class Qwen3ASRTokenizer(Qwen3Tokenizer):
    """Tokenizer for Qwen3-ASR models.

    This tokenizer inherits from `Qwen3Tokenizer` but is bound to
    `Qwen3ASRBackbone` to enable accessing presets under the `qwen3_asr`
    namespace.
    """

    backbone_cls = Qwen3ASRBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        self._add_special_token("<|AUDIO|>", "audio_token")
        self._add_special_token("<asr_text>", "asr_token")
        self._add_special_token("<|im_start|>", "start_token")
        # [time] is used by the Forced Aligner part of the Qwen3-ASR family.
        self._add_special_token("[time]", "time_token")
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

    @property
    def audio_token_id(self):
        return self.special_token_to_id("<|AUDIO|>")

    @property
    def asr_token_id(self):
        return self.special_token_to_id("<asr_text>")

    @property
    def time_token_id(self):
        return self.special_token_to_id("[time]")
