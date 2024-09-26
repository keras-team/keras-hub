import json

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


def _load_dict(dict_or_path):
    if isinstance(dict_or_path, str):
        with open(dict_or_path, "r", encoding="utf-8") as f:
            dict_or_path = json.load(f)
    return dict_or_path


@keras_hub_export(
    [
        "keras_hub.tokenizers.WhisperTokenizer",
        "keras_hub.models.WhisperTokenizer",
    ]
)
class WhisperTokenizer(BytePairTokenizer):
    """Whisper text tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`.
    This tokenizer does not provide truncation or padding of inputs.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
        special_tokens: string or dict, maps special tokens to integer IDs. If
            it is a string, it should be the path to a JSON file.
        language_tokens: string or dict, maps language tokens to integer IDs. If
            not None, the tokenizer will be assumed to be a multilingual
            tokenizer.
    """

    backbone_cls = WhisperBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        special_tokens=None,
        language_tokens=None,
        **kwargs,
    ):
        special_tokens = _load_dict(special_tokens)
        if language_tokens is not None:
            language_tokens = _load_dict(language_tokens)

        # Necessary special tokens.
        self.bos_token = "<|startoftranscript|>"
        self.eos_token = "<|endoftext|>"
        # TODO: The pad token for the multilingual tokenizer is actually
        # "", but it errors out (OOM). After BPE is fixed, we can update
        # this to "". For now, we will use `"<|endoftext|>"`.
        self.pad_token = "<|endoftext|>"

        self.no_timestamps_token = "<|notimestamps|>"
        # Task special tokens.
        self.translate_token = "<|translate|>"
        self.transcribe_token = "<|transcribe|>"

        for token in [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.no_timestamps_token,
            self.translate_token,
            self.transcribe_token,
        ]:
            if token not in special_tokens:
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`special_tokens`. Please provide `'{token}'` in your "
                    "`special_tokens`."
                )

        self.bos_token_id = special_tokens[self.bos_token]
        self.eos_token_id = special_tokens[self.eos_token]
        self.pad_token_id = special_tokens[self.pad_token]
        self.no_timestamps_token_id = special_tokens[self.no_timestamps_token]
        self.translate_token_id = special_tokens[self.translate_token]
        self.transcribe_token_id = special_tokens[self.transcribe_token]

        self._special_token_dict = special_tokens
        self.language_tokens = language_tokens
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

    @property
    def special_tokens(self):
        return list(self._special_token_dict.keys())

    @property
    def special_token_ids(self):
        return list(self._special_token_dict.values())

    def save_assets(self, dir_path):
        # TODO: whisper is currently mutating it's vocabulary before passing
        # it to the super class, so we need to restore the unmutated vocabulary
        # before saving our assets. We should find a more robust (and memory
        # efficient) way to do this.
        vocabulary = self.vocabulary
        self.vocabulary = self._initial_vocabulary
        super().save_assets(dir_path)
        self.vocabulary = vocabulary

    def set_vocabulary_and_merges(self, vocabulary, merges):
        if vocabulary is not None:
            vocabulary = _load_dict(vocabulary)
            self._initial_vocabulary = dict(vocabulary)

            if self.language_tokens is not None:
                # Multilingual tokenizer.
                # Add language tokens to the vocabulary. This makes
                # detokenization easier for us.
                vocabulary = {
                    **vocabulary,
                    **self.language_tokens,
                }

            for token in [
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.no_timestamps_token,
                self.translate_token,
                self.transcribe_token,
            ]:
                vocabulary[token] = self._special_token_dict[token]
        else:
            self._initial_vocabulary = None

        super().set_vocabulary_and_merges(vocabulary, merges)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "special_tokens": self._special_token_dict,
                "language_tokens": self.language_tokens,
            }
        )
        return config
