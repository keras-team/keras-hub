from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.GptOssTokenizer",
        "keras_hub.models.GptOssTokenizer",
    ]
)
class GptOssTokenizer(BytePairTokenizer):
    """A GptOss tokenizer using BytePair encoding.

    Tokenizer is a subclass of `keras_hub.tokenizers.BytePairTokenizer`.
    It uses a BytePair encoding model to tokenize strings. It also adds special
    tokens for the start and end of a sequence.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules.
    """

    backbone_cls = GptOssBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        """Initializes the GptOssTokenizer.

        Args:
            vocabulary: string or dict, maps token to integer ids.
            merges: string or list, contains the merge rule.
            **kwargs: Additional keyword arguments.
        """
        self.start_token_id = None
        self._add_special_token("<|endoftext|>", "end_token")
        self.pad_token_id = 0
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)
