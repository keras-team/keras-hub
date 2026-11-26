from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.utils.tensor_utils import canonicalize_python_token_inputs
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen3ASRTokenizer",
        "keras_hub.models.Qwen3ASRTokenizer",
    ]
)
class Qwen3ASRTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3-ASR models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3-ASR models,
    handling special tokens like EOS (end of sequence) and audio placeholders.

    Args:
        vocabulary: dict or str, optional. Dictionary mapping tokens to token
            IDs, or path to vocabulary file. Defaults to `None`.
        merges: list or str, optional. List of BPE merges, or path to merges
            file. Defaults to `None`.
        unsplittable_tokens: list of str, optional. Tokens that should never be
            split during tokenization. Required Qwen3-ASR special tokens are
            automatically included. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to `BytePairTokenizer`.
    """

    backbone_cls = Qwen3ASRBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        unsplittable_tokens=None,
        **kwargs,
    ):
        self._add_special_token("<|im_end|>", "end_token")
        self._add_special_token("<|endoftext|>", "pad_token")

        required_tokens = [
            *_SPECIAL_TOKENS,
            "<asr_text>",
        ]
        unsplittable_tokens = sorted(
            set(unsplittable_tokens or ()) | set(required_tokens)
        )

        if isinstance(vocabulary, dict):
            missing = [tok for tok in required_tokens if tok not in vocabulary]
            if missing:
                raise ValueError(
                    f"Missing required special tokens in vocabulary: {missing}."
                )

        self.start_token_id = None
        self.start_token = None

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=unsplittable_tokens,
            **kwargs,
        )

        if self.vocabulary is not None:
            missing = [
                tok for tok in required_tokens if tok not in self.vocabulary
            ]
            if missing:
                raise ValueError(
                    f"Missing required special tokens in vocabulary: {missing}."
                )

    def _special_token_ids(self):
        special_ids = {
            self.vocabulary[tok]
            for tok in _SPECIAL_TOKENS
            if tok in self.vocabulary
        }
        special_ids |= set(self.special_token_ids)
        asr_text_id = self.vocabulary.get("<asr_text>")
        if asr_text_id is not None:
            special_ids.discard(asr_text_id)
        return special_ids

    @preprocessing_function
    def _detokenize_tf(self, inputs, skip_special_tokens=False):
        if not skip_special_tokens:
            return super()._detokenize_tf(inputs)
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        special_ids = self._special_token_ids()
        if special_ids:
            special_tensor = tf.constant(
                sorted(special_ids), dtype=inputs.dtype
            )
            mask = tf.reduce_all(
                tf.not_equal(tf.expand_dims(inputs, -1), special_tensor),
                axis=-1,
            )
            inputs = tf.ragged.boolean_mask(inputs, mask)
        outputs = super()._detokenize_tf(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def detokenize(self, inputs, skip_special_tokens=False):
        """Convert a list of integer ids to a string."""
        self._check_vocabulary()
        if self._use_tf_workflow():
            return self._detokenize_tf(
                inputs, skip_special_tokens=skip_special_tokens
            )
        if skip_special_tokens:
            self._maybe_initialized_tokenizers()
            sequences, batched = canonicalize_python_token_inputs(inputs)
            special_ids = self._special_token_ids()
            sequences = [
                [tid for tid in seq if tid not in special_ids]
                for seq in sequences
            ]
            outputs = self._tokenizer.decode_batch(
                sequences, skip_special_tokens=False
            )
            return outputs if batched else outputs[0]
        return super().detokenize(inputs)
