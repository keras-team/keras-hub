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
        # We define unsplittable tokens here and pass them down.
        # This ensures BytePairTokenizer registers them correctly in the regex.
        unsplittable_tokens = kwargs.get("unsplittable_tokens", None)
        if unsplittable_tokens is None:
            unsplittable_tokens = [
                "<|AUDIO|>",
                "<asr_text>",
                "<|im_start|>",
                "<|im_end|>",
                "[time]",
                "<|audio_start|>",
                "<|audio_end|>",
                "<|endoftext|>",
            ]
        kwargs["unsplittable_tokens"] = unsplittable_tokens

        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

        # Re-add/Verify special tokens to set properties correctly.
        # We use internal names to avoid collision with Qwen3Tokenizer.
        self._add_special_token("<|AUDIO|>", "audio_token")
        self._add_special_token("<asr_text>", "asr_token")
        self._add_special_token("<|im_start|>", "start_token")
        self._add_special_token("<|im_end|>", "end_token")
        self._add_special_token("[time]", "time_token")
        self._add_special_token("<|audio_start|>", "audio_start_token")
        self._add_special_token("<|audio_end|>", "audio_end_token")
