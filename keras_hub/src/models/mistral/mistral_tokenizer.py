try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tokenizers as hf_tokenizers
    from tokenizers import decoders
    from tokenizers import models as hf_models
    from tokenizers import pre_tokenizers
except ImportError:
    hf_tokenizers = None


@keras_hub_export(
    [
        "keras_hub.tokenizers.MistralTokenizer",
        "keras_hub.models.MistralTokenizer",
    ]
)
class MistralTokenizer(SentencePieceTokenizer):
    """Mistral tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Mistral models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Mistral preset.

    This tokenizer is used by SentencePiece-based Mistral presets. Presets that
    ship a Tekken (byte-level BPE) vocabulary, such as Magistral, use
    `keras_hub.tokenizers.MistralTekkenTokenizer` instead.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.MistralTokenizer.from_preset(
        "mistral_7b_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = MistralBackbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self.pad_token_id = 0
        super().__init__(proto=proto, **kwargs)


@keras_hub_export(
    [
        "keras_hub.tokenizers.MistralTekkenTokenizer",
        "keras_hub.models.MistralTekkenTokenizer",
    ]
)
class MistralTekkenTokenizer(BytePairTokenizer):
    """Mistral Tekken tokenizer layer based on byte-level BPE.

    This tokenizer class handles Mistral's Tekken (tiktoken-style byte-level
    BPE) vocabulary, used by presets such as Magistral. It is based on
    `keras_hub.tokenizers.BytePairTokenizer`, but uses the Tekken
    pre-tokenization regex instead of the GPT-2/Llama3 pattern hardcoded in the
    base class, and checks for the special tokens needed by Mistral models.

    The `vocabulary` and `merges` are usually produced from a `tekken.json`
    file by the Hugging Face conversion path; see
    `keras_hub.src.utils.transformers.convert_mistral`.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: A dict mapping token strings to integer ids, or a path to a
            vocabulary JSON file.
        merges: A list of BPE merge rules, or a path to a merges file.
        split_pattern: str. The Tekken pre-tokenization regex.

    Examples:
    ```python
    tokenizer = keras_hub.models.MistralTekkenTokenizer.from_preset(
        "hf://mistralai/Magistral-Small-2506",
    )
    tokenizer("The quick brown fox jumped.")
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = MistralBackbone

    def __init__(
        self, vocabulary=None, merges=None, split_pattern=None, **kwargs
    ):
        self.split_pattern = split_pattern
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=[self.start_token, self.end_token],
            **kwargs,
        )

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        self.vocabulary = vocabulary.copy()
        self.merges = list(merges)
        _merges = []
        for merge in self.merges:
            if "#version:" in merge.lstrip():
                continue
            a, b = str(merge).split(" ")
            _merges.append((a, b))
        self._tokenizer = hf_tokenizers.Tokenizer(
            hf_models.BPE(vocab=vocabulary, merges=_merges, fuse_unk=False)
        )
        if self.unsplittable_tokens:
            self._tokenizer.add_special_tokens(self.unsplittable_tokens)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    hf_tokenizers.Regex(self.split_pattern),
                    behavior="isolated",
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space, use_regex=False
                ),
            ]
        )
        self._tokenizer.decoder = decoders.ByteLevel()

        # Dummy attrs for serialization compatibility with the base class.
        if not hasattr(self, "cache"):
            self.byte2unicode = None
            self.unicode2byte = None
            self.cache = None
            self.id_to_token_map = None
            self.token_to_id_map = None
            self.merge_ranks_lookup_default = None
            self.merge_ranks = None

    def _set_vocabulary_and_merges_tf(self, vocabulary, merges):
        # The base class hardcodes the GPT-2 split regex in its `tf.data`
        # path, which does not match Tekken. We instead bridge to the
        # `tokenizers` backend from within the graph (see `_tokenize_tf`), so
        # there is nothing to build here.
        self.vocabulary = vocabulary.copy()
        self.merges = list(merges)

    def _maybe_initialized_tokenizers(self):
        if getattr(self, "_tokenizer", None) is None:
            self._set_vocabulary_and_merges_tokenizers(
                self.vocabulary, self.merges
            )

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tokenizers()

        def _encode(string_tensor):
            values = string_tensor.numpy()
            strings = [v.decode("utf-8") for v in values.tolist()]
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
                shape=output_shape,
                default_value=getattr(self, "pad_token_id", 0),
            )
        if unbatched:
            tokens = tokens[0]
        return tokens

    def get_config(self):
        config = super().get_config()
        config.update({"split_pattern": self.split_pattern})
        # `unsplittable_tokens` is derived from the special tokens in the
        # constructor, so it is not a separate config argument.
        del config["unsplittable_tokens"]
        return config
