import tokenizers
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

# Copied from the `pre_tokenizer` entry of the checkpoint's `tokenizer.json`.
# The pattern splits camelCase words and keeps combining marks (`\p{M}`) inside
# words. The shared Llama-3 pattern does neither.
SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.MuseGlimmerTokenizer",
        "keras_hub.models.MuseGlimmerTokenizer",
    ]
)
class MuseGlimmerTokenizer(BytePairTokenizer):
    """A MuseGlimmer byte-level BPE tokenizer.

    Uses the regex pre-tokenizer split from the checkpoint's
    `tokenizer.json`. The split separates camelCase words and keeps combining
    marks inside words. `image_token_id`/`video_token_id` are fixed
    placeholder ids (per `config.json`) that the preprocessor expands to
    per-patch image/video feature positions — they are not resolved from a
    named special-token string, since the checkpoint assigns them directly
    by id among its ~2,043 reserved special tokens.

    Args:
        vocabulary: dict or None. Maps token strings to integer ids.
        merges: list or None. BPE merge rules.
        bos_token: str. Beginning-of-sequence token.
        eos_token: str. End-of-sequence token.
        pad_token: str. Padding token.
        eot_token: str or `None`. End-of-turn token. `generate()` stops at
            this token and at `eos_token`. Defaults to `"<|eot|>"`.
        image_token_id: int. Placeholder token id expanded to per-patch
            image embeddings by the preprocessor. Defaults to `200092`.
        video_token_id: int. Placeholder token id expanded to per-frame
            video embeddings by the preprocessor. Defaults to `200091`.
    """

    backbone_cls = MuseGlimmerBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|finetune_right_pad|>",
        eot_token="<|eot|>",
        image_token_id=200092,
        video_token_id=200091,
        **kwargs,
    ):
        self._add_special_token(bos_token, "start_token")
        self._add_special_token(eos_token, "end_token")
        self._add_special_token(pad_token, "pad_token")
        self.eot_token = eot_token
        if eot_token is not None:
            self._add_special_token(eot_token, "end_token2")
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        super()._set_vocabulary_and_merges_tokenizers(vocabulary, merges)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    tokenizers.Regex(SPLIT_PATTERN), behavior="isolated"
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space, use_regex=False
                ),
            ]
        )

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        # The base class splits text with the GPT-2 regex and loops over every
        # special token in the graph. Both fail for this checkpoint. This
        # method calls the `tokenizers` backend from the graph instead.
        self._maybe_initialized_tokenizers()

        def _encode(string_tensor):
            values = string_tensor.numpy().tolist()
            strings = [v.decode("utf-8") for v in values]
            encodings = self._tokenizer.encode_batch(
                strings, add_special_tokens=False
            )
            return tf.ragged.constant(
                [e.ids for e in encodings], dtype=self.compute_dtype
            )

        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)
        tokens = tf.py_function(
            _encode,
            [inputs],
            Tout=tf.RaggedTensorSpec(
                shape=[None, None],
                dtype=self.compute_dtype,
                ragged_rank=1,
            ),
        )
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(
                shape=output_shape, default_value=self.pad_token_id
            )
        if unbatched:
            tokens = tokens[0]
        return tokens

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "eot_token": self.eot_token,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
            }
        )
        return config
