import tensorflow as tf
from keras import layers
from keras.src.saving import register_keras_serializable
from ...tokenizers.word_piece_tokenizer import WordPieceTokenizer

@register_keras_serializable()
class LayoutLMv3Tokenizer(WordPieceTokenizer):
    """LayoutLMv3 tokenizer.
    
    This tokenizer inherits from WordPieceTokenizer and adds LayoutLMv3-specific
    special tokens and functionality.
    
    Args:
        vocabulary: A list of strings containing the vocabulary.
        lowercase: Whether to lowercase the input text.
        strip_accents: Whether to strip accents from the input text.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        vocabulary=None,
        lowercase=True,
        strip_accents=True,
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            strip_accents=strip_accents,
            **kwargs,
        )
        
        # Special tokens
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        
        # Special token IDs
        self.cls_token_id = self.token_to_id(self.cls_token)
        self.sep_token_id = self.token_to_id(self.sep_token)
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.mask_token_id = self.token_to_id(self.mask_token)
        self.unk_token_id = self.token_to_id(self.unk_token)
        
        # Special token masks
        self.cls_token_mask = tf.constant(1, dtype=tf.int32)
        self.sep_token_mask = tf.constant(1, dtype=tf.int32)
        self.pad_token_mask = tf.constant(0, dtype=tf.int32)
        self.mask_token_mask = tf.constant(1, dtype=tf.int32)
        self.unk_token_mask = tf.constant(1, dtype=tf.int32)
    
    def call(self, inputs):
        """Tokenize the input text.
        
        Args:
            inputs: A string or list of strings to tokenize.
            
        Returns:
            A dictionary containing:
                - token_ids: The token IDs.
                - padding_mask: The padding mask.
                - attention_mask: The attention mask.
        """
        # Tokenize the input text
        tokenized = super().call(inputs)
        
        # Add special tokens
        token_ids = tokenized["token_ids"]
        padding_mask = tokenized["padding_mask"]
        
        # Add [CLS] token at the beginning
        cls_token_ids = tf.fill([tf.shape(token_ids)[0], 1], self.cls_token_id)
        cls_token_mask = tf.fill([tf.shape(padding_mask)[0], 1], self.cls_token_mask)
        
        token_ids = tf.concat([cls_token_ids, token_ids], axis=1)
        padding_mask = tf.concat([cls_token_mask, padding_mask], axis=1)
        
        # Add [SEP] token at the end
        sep_token_ids = tf.fill([tf.shape(token_ids)[0], 1], self.sep_token_id)
        sep_token_mask = tf.fill([tf.shape(padding_mask)[0], 1], self.sep_token_mask)
        
        token_ids = tf.concat([token_ids, sep_token_ids], axis=1)
        padding_mask = tf.concat([padding_mask, sep_token_mask], axis=1)
        
        # Create attention mask
        attention_mask = tf.cast(padding_mask, dtype=tf.int32)
        
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "attention_mask": attention_mask,
        }
    
    def detokenize(self, token_ids):
        """Convert token IDs back to text.
        
        Args:
            token_ids: A tensor of token IDs.
            
        Returns:
            A list of strings containing the detokenized text.
        """
        # Remove special tokens
        token_ids = token_ids[:, 1:-1]  # Remove [CLS] and [SEP]
        
        # Convert to text
        return super().detokenize(token_ids)
    
    def get_config(self):
        """Get the tokenizer configuration.
        
        Returns:
            A dictionary containing the tokenizer configuration.
        """
        config = super().get_config()
        config.update({
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "unk_token": self.unk_token,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create a tokenizer from a configuration dictionary.
        
        Args:
            config: A dictionary containing the tokenizer configuration.
            
        Returns:
            A LayoutLMv3Tokenizer instance.
        """
        return cls(**config) 