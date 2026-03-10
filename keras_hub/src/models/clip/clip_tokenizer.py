import tokenizers
from tokenizers import decoders
from tokenizers import models
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import processors

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.clip.clip_backbone import CLIPBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.byte_pair_tokenizer import split_strings_for_bpe
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export(
    [
        "keras_hub.tokenizers.CLIPTokenizer",
        "keras_hub.models.CLIPTokenizer",
    ]
)
class CLIPTokenizer(BytePairTokenizer):
    """A CLIP tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by CLIP
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a CLIP preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
        pad_with_end_token: bool. Whether to pad the output with `end_token`.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.CLIPTokenizer.from_preset(
        "clip_vit_base_patch32"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = CLIPBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        pad_with_end_token=False,
        **kwargs,
    ):
        self._add_special_token("<|startoftext|>", "start_token")
        self._add_special_token("<|endoftext|>", "end_token")
        self.pad_token_id = 0
        self.pad_with_end_token = pad_with_end_token

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=[self.start_token, self.end_token],
            **kwargs,
        )

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        # CLIPTokenizer has the extra settings.
        # Ref: transformers.models.clip.tokenization_clip
        vocabulary = self.vocabulary.copy()
        merges = self.merges
        _merges = []
        for merge in merges:
            if "#version:" in merge.lstrip():
                continue
            a, b = str(merge).split(" ")
            if a not in vocabulary or b not in vocabulary:
                raise ValueError(
                    f"Merge rule '{merge}' contains token '{a}' or '{b}' that "
                    "is not in the vocabulary."
                )
            _merges.append((a, b))
        self._tokenizer = tokenizers.Tokenizer(
            models.BPE(
                vocab=vocabulary,
                merges=_merges,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token="<|endoftext|>",
            )
        )
        if self.unsplittable_tokens:
            self._tokenizer.add_special_tokens(self.unsplittable_tokens)
        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.NFC(),
                normalizers.Replace(tokenizers.Regex(r"\s+"), " "),
                normalizers.Lowercase(),
            ]
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    tokenizers.Regex(
                        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
                    ),
                    behavior="removed",
                    invert=True,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space
                ),
            ]
        )
        self._tokenizer.decoder = decoders.ByteLevel()

        # Dummy attrs for serialization compatibility.
        if not hasattr(self, "cache"):
            self.byte2unicode = None
            self.unicode2byte = None
            self.cache = None
            self.id_to_token_map = None
            self.token_to_id_map = None
            self.merge_ranks_lookup_default = None
            self.merge_ranks = None

    def set_vocabulary_and_merges(self, vocabulary, merges):
        super().set_vocabulary_and_merges(vocabulary, merges)
        if self.pad_with_end_token:
            self.pad_token_id = self.end_token_id
        if getattr(self, "_tokenizer") is not None:
            self._tokenizer.post_processor = processors.RobertaProcessing(
                sep=(str(self.end_token), self.end_token_id),
                cls=(str(self.start_token), self.start_token_id),
                add_prefix_space=False,
                trim_offsets=False,
            )

    def _bpe_merge_and_update_cache_tf(self, tokens):
        """Process unseen tokens and add to cache."""

        def _transform_bytes(tokens):
            """Map token bytes to unicode using `byte2unicode`."""
            split_bytes = tf.strings.bytes_split(tokens)
            split_unicode = self.byte2unicode.lookup(split_bytes)
            return split_unicode

        words = _transform_bytes(tokens)

        # In CLIP, we need to add `</w>` to the last word.
        words = tf.strings.reduce_join(words, axis=1, separator=" ")
        words = tf.strings.join([words, "</w>"])
        words = tf.strings.split(words, sep=" ")
        tokenized_words = self._bpe_merge_tf(words)

        # For each word, join all its token by a whitespace,
        # e.g., ["dragon", "fly"] => "dragon fly" for hash purpose.
        tokenized_words = tf.strings.reduce_join(
            tokenized_words, axis=1, separator=" "
        )
        self.cache.insert(tokens, tokenized_words)

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        if self.add_prefix_space:
            inputs = tf.strings.join([" ", inputs])

        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)
        if inputs.shape.rank > 1:
            raise ValueError(
                "`tokenize()` inputs should be a string, list of strings, or "
                f"string tensor with rank < 2. Received: {inputs}"
            )

        raw_tokens = split_strings_for_bpe(inputs, self.unsplittable_tokens)

        # Strip and remove empty tokens.
        raw_tokens = tf.strings.strip(raw_tokens)
        raw_tokens = tf.ragged.boolean_mask(raw_tokens, raw_tokens != "")
        token_row_splits = raw_tokens.row_splits
        flat_tokens = raw_tokens.flat_values

        # Check cache.
        cache_lookup = self.cache.lookup(flat_tokens)
        cache_mask = cache_lookup == ""
        has_unseen_words = tf.math.reduce_any(
            (cache_lookup == "") & (flat_tokens != "")
        )

        def process_unseen_tokens():
            unseen_tokens = tf.boolean_mask(flat_tokens, cache_mask)
            self._bpe_merge_and_update_cache_tf(unseen_tokens)
            return self.cache.lookup(flat_tokens)

        # If `has_unseen_words == True`, it means not all tokens are in cache,
        # we will process the unseen tokens. Otherwise return the cache lookup.
        tokenized_words = tf.cond(
            has_unseen_words,
            process_unseen_tokens,
            lambda: cache_lookup,
        )
        tokens = tf.strings.split(tokenized_words, sep=" ")
        if self.compute_dtype != tf.string:
            # Encode merged tokens.
            tokens = self.token_to_id_map.lookup(tokens)

        # Unflatten to match input.
        tokens = tf.RaggedTensor.from_row_splits(
            tokens.flat_values,
            tf.gather(tokens.row_splits, token_row_splits),
        )

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        # Convert to a dense output if input in scalar
        if unbatched:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])
        return tokens

    def _tokenize_tokenizers(self, inputs):
        outputs = super()._tokenize_tokenizers(inputs)
        is_batched = True
        if isinstance(outputs, str):
            is_batched = False
            outputs = [outputs]
        elif isinstance(outputs, list) and isinstance(outputs[0], int):
            is_batched = False
            outputs = [outputs]
        outputs = [output[1:-1] for output in outputs]
        if not is_batched:
            outputs = outputs[0]
        return outputs

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        self._check_vocabulary()
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        inputs = tf.cast(inputs, self.dtype)
        unicode_text = tf.strings.reduce_join(
            self.id_to_token_map.lookup(inputs), axis=-1
        )

        # When detokenizing, we need to remove </w> and extra whitespace.
        unicode_text = tf.strings.regex_replace(unicode_text, r"</w>", " ")
        unicode_text = tf.strings.strip(unicode_text)

        split_unicode_text = tf.strings.unicode_split(unicode_text, "UTF-8")
        outputs = tf.strings.reduce_join(
            self.unicode2byte.lookup(split_unicode_text), axis=-1
        )

        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def _detokenize_tokenizers(self, inputs):
        outputs = super()._detokenize_tokenizers(inputs)

        def _remove_special_token(inputs):
            is_batched = True
            if isinstance(inputs, str):
                inputs = [inputs]
                is_batched = False
            for special_token in ("</w>", self.start_token, self.end_token):
                inputs = [
                    input.replace(str(special_token), "") for input in inputs
                ]
            if not is_batched:
                inputs = inputs[0]
            return inputs

        return _remove_special_token(outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pad_with_end_token": self.pad_with_end_token,
            }
        )
        # In the constructor, we pass the list of special tokens to the
        # `unsplittable_tokens` arg of the superclass' constructor. Hence, we
        # delete it from the config here.
        del config["unsplittable_tokens"]
        return config
