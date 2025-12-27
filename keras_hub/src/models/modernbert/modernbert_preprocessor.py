import keras
from keras import ops
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing import start_end_packer

@keras_hub_export("keras_hub.models.ModernBertPreprocessor")
class ModernBertPreprocessor(Preprocessor):
    """ModernBERT preprocessor.

    This class provides the preprocessing logic for ModernBERT, including
    tokenization and packing sequences with a padding mask.

    Args:
        tokenizer: A `keras_hub.models.ModernBertTokenizer` instance.
        sequence_length: int. The length of the packed sequence.
        truncate: str. The algorithm for truncating a sequence.
        **kwargs: Standard `Preprocessor` arguments.
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        
        self.packer = start_end_packer.StartEndPacker(
            start_token_id=tokenizer.cls_token_id,
            end_token_id=tokenizer.sep_token_id,
            pad_value=tokenizer.pad_token_id,
            sequence_length=sequence_length,
            truncate=truncate,
        )

    def call(self, x, training=None):
        """Preprocesses the input text.
        
        Args:
            x: A string, list of strings, or a Tensor of strings.
        """
        token_ids = self.tokenizer(x)
        
        token_ids = self.packer(token_ids)
        
        # Create the padding mask (1 for real tokens, 0 for padding)
        padding_mask = ops.not_equal(token_ids, self.tokenizer.pad_token_id)
        
        return {
            "token_ids": ops.cast(token_ids, "int32"),
            "padding_mask": ops.cast(padding_mask, "int32"),
        }

    @property
    def tokenizer(self):
        """The tokenizer used by this preprocessor."""
        return self.layers[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.packer.sequence_length,
        })
        return config