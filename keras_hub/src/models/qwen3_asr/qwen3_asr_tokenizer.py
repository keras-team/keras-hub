import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen3ASRTokenizer",
        "keras_hub.models.Qwen3ASRTokenizer",
    ]
)
class Qwen3ASRTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3-ASR models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3-ASR models,
    handling special tokens like BOS (beginning of sequence) and EOS (end of
    sequence).

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
        bos_token: Beginning of sequence token. Defaults to None.
        eos_token: End of sequence token. Defaults to "<|endoftext|>".
        misc_special_tokens: Set of additional special tokens. Defaults to
            empty set.
    """

    backbone_cls = Qwen3ASRBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # Add EOS token
        self._add_special_token("<|endoftext|>", "end_token")
        self._add_special_token("<|im_end|>", "end_token2")

        pad_token = "<|endoftext|>"
        self._add_special_token(pad_token, "pad_token")

        required_tokens = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
            "<asr_text>",
        ]
        if "unsplittable_tokens" in kwargs:
            kwargs["unsplittable_tokens"] = sorted(
                list(set(kwargs["unsplittable_tokens"]) | set(required_tokens))
            )
        else:
            kwargs["unsplittable_tokens"] = sorted(required_tokens)

        self.start_token_id = None
        self.start_token = None

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

    def detokenize(self, inputs, skip_special_tokens=False):
        """Convert a list of integer ids to a string."""
        if skip_special_tokens:
            self._maybe_initialized_tokenizers()
            # Tokenizers library decode supports skipping special tokens.
            # Handle both single sequence and batch
            inputs_np = np.array(inputs)
            if len(inputs_np.shape) == 1:
                res = self._tokenizer.decode(
                    inputs_np.tolist(), skip_special_tokens=True
                )
                return res
            else:
                res = self._tokenizer.decode_batch(
                    inputs_np.tolist(), skip_special_tokens=True
                )
                return res
        return super().detokenize(inputs)
