import keras
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras.utils.register_keras_serializable(package="keras_hub")
@keras_hub_export("keras_hub.models.ModernBertTokenizer")
class ModernBertTokenizer(BytePairTokenizer):
    """ModernBERT tokenizer based on Byte-Pair Encoding.

    ModernBERT uses a byte-level BPE tokenizer (OLMo style). This tokenizer
    is responsible for converting raw strings into integer token IDs.

    Args:
        vocabulary: string or dict. A path to a BPE vocabulary file or a dict.
        merges: string or list. A path to a BPE merges file or a list.
        **kwargs: Standard `BytePairTokenizer` arguments.

    """

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

        self.pad_token_id = 1
        self.bos_token_id = 50279
        self.eos_token_id = 50279
        self.cls_token_id = 50279
        self.sep_token_id = 50279
        self.mask_token_id = 50284
    
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocabulary": self.vocabulary,
            "merges": self.merges,
        })
        return config