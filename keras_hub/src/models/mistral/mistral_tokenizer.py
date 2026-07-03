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


class _MistralTekkenTokenizer(BytePairTokenizer):
    """Byte-level BPE backend for Mistral's Tekken tokenizer.

    Tekken (used by e.g. Magistral) is a tiktoken-style byte-level BPE
    tokenizer. It differs from the GPT-2/Llama3 style handled by the base
    `BytePairTokenizer` in its pre-tokenization regex, so we override the
    `tokenizers` backend to use the Tekken split pattern and bridge it into
    the `tf.data` graph path with a `tf.py_function`.
    """

    def __init__(
        self, vocabulary=None, merges=None, split_pattern=None, **kwargs
    ):
        self.split_pattern = split_pattern
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

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

    def _maybe_initialized_tokenizers(self):
        if getattr(self, "_tokenizer", None) is None:
            self._set_vocabulary_and_merges_tokenizers(
                self.vocabulary, self.merges
            )

    def get_config(self):
        config = super().get_config()
        config.update({"split_pattern": self.split_pattern})
        return config


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

    Most Mistral presets use a SentencePiece vocabulary. Newer presets such as
    Magistral instead ship a Tekken (byte-level BPE) vocabulary; these are
    handled transparently by passing `vocabulary` and `merges` instead of a
    `proto`, and tokenization is delegated to an internal
    `keras_hub.tokenizers.BytePairTokenizer`.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format. Provide this for SentencePiece
            presets.
        vocabulary: Optional. A dict mapping token strings to integer ids, or a
            path to a vocabulary JSON file. Provide this together with `merges`
            for Tekken (byte-level BPE) presets.
        merges: Optional. A list of BPE merge rules, or a path to a merges
            file. Provide this together with `vocabulary` for Tekken presets.
        split_pattern: Optional. The pre-tokenization regex used by the Tekken
            backend. Only used when `vocabulary`/`merges` are provided.

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

    def __init__(
        self,
        proto=None,
        vocabulary=None,
        merges=None,
        split_pattern=None,
        sequence_length=None,
        dtype="int32",
        **kwargs,
    ):
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self.pad_token_id = 0

        # A Tekken (byte-level BPE) tokenizer is selected when a `proto` is not
        # used. `split_pattern` is always present for Tekken presets (including
        # after deserialization, when `vocabulary`/`merges` arrive later as
        # assets), so it is the reliable discriminator.
        self._is_tekken = (
            vocabulary is not None
            or merges is not None
            or split_pattern is not None
        )
        if not self._is_tekken:
            super().__init__(
                proto=proto,
                sequence_length=sequence_length,
                dtype=dtype,
                **kwargs,
            )
            return

        # Delegate tokenization to an internal `BytePairTokenizer` while
        # remaining a `MistralTokenizer` instance so the preset/preprocessor
        # machinery keeps working.
        self._tekken_split_pattern = split_pattern
        self._bpe = _MistralTekkenTokenizer(
            vocabulary=vocabulary,
            merges=merges,
            split_pattern=split_pattern,
            unsplittable_tokens=[self.start_token, self.end_token],
            dtype=dtype,
            sequence_length=sequence_length,
        )
        # Bypass the SentencePiece `__init__`, which requires a proto.
        from keras_hub.src.tokenizers.tokenizer import Tokenizer

        Tokenizer.__init__(self, dtype=dtype, **kwargs)
        self.file_assets = self._bpe.file_assets
        # The vocabulary may not be available yet (it arrives via `load_assets`
        # during deserialization); only resolve special-token ids once it is.
        if vocabulary is not None:
            self._update_special_token_ids()

    # --- Tekken delegation ---------------------------------------------------

    def save_assets(self, dir_path):
        if self._is_tekken:
            return self._bpe.save_assets(dir_path)
        return super().save_assets(dir_path)

    def load_assets(self, dir_path):
        if self._is_tekken:
            self._bpe.load_assets(dir_path)
            self._update_special_token_ids()
            return
        return super().load_assets(dir_path)

    def vocabulary_size(self):
        if self._is_tekken:
            return self._bpe.vocabulary_size()
        return super().vocabulary_size()

    def get_vocabulary(self):
        if self._is_tekken:
            return self._bpe.get_vocabulary()
        return super().get_vocabulary()

    def id_to_token(self, id):
        if self._is_tekken:
            return self._bpe.id_to_token(id)
        return super().id_to_token(id)

    def token_to_id(self, token):
        if self._is_tekken:
            return self._bpe.token_to_id(token)
        return super().token_to_id(token)

    def tokenize(self, inputs):
        if self._is_tekken:
            return self._bpe.tokenize(inputs)
        return super().tokenize(inputs)

    def detokenize(self, inputs):
        if self._is_tekken:
            return self._bpe.detokenize(inputs)
        return super().detokenize(inputs)

    def compute_output_spec(self, input_spec):
        if self._is_tekken:
            return self._bpe.compute_output_spec(input_spec)
        return super().compute_output_spec(input_spec)

    def get_config(self):
        if not self._is_tekken:
            return super().get_config()
        # Skip `SentencePieceTokenizer.get_config`, which emits proto-specific
        # keys. Rebuild the config from the base `Tokenizer` plus the Tekken
        # arguments (vocabulary/merges are saved as assets).
        from keras_hub.src.tokenizers.tokenizer import Tokenizer

        config = Tokenizer.get_config(self)
        config.update(
            {
                "proto": None,
                "vocabulary": None,
                "merges": None,
                "split_pattern": self._tekken_split_pattern,
                "sequence_length": self._bpe.sequence_length,
            }
        )
        return config
