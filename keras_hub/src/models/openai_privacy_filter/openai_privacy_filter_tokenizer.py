from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.OpenAIPrivacyFilterTokenizer",
        "keras_hub.models.OpenAIPrivacyFilterTokenizer",
    ]
)
class OpenAIPrivacyFilterTokenizer(BytePairTokenizer):
    """An OpenAI Privacy Filter tokenizer using Byte-Pair Encoding.

    This tokenizer layer provides an implementation of a BPE tokenizer
    for the OpenAI Privacy Filter model.

    Args:
        vocabulary: string or dict, maps token to integer ids.
        merges: string or list, contains the merge rule.
    """

    backbone_cls = OpenAIPrivacyFilterBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        self.start_token_id = None
        self.pad_token_id = 199999
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)
