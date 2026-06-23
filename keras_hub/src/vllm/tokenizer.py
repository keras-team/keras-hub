"""
Tokenizer adapter for bridging KerasHub Tokenizers to vLLM.

This module provides the necessary wrappers to ensure vLLM processes tokens
identically to the native KerasHub environment.
"""

from typing import Any, Dict, List

from keras_hub import models


class KerasVLLMTokenizerAdapter:
    """Adapts a Keras Hub Tokenizer to the interface expected by vLLM.
    
    To ensure complete numerical and logical consistency, the vLLM runner 
    uses the exact tokenizer assets provided by the KerasHub preset.
    """

    def __init__(self, preset_name: str):
        """Initializes the tokenizer wrapper for vLLM.
        
        Args:
            preset_name: The name of the KerasHub preset to load.
        """
        self.preset_name = preset_name
        self.tokenizer = models.Tokenizer.from_preset(preset_name)

    @property
    def is_fast(self) -> bool:
        """Returns whether this is a fast tokenizer."""
        return False

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.tokenizer.vocabulary_size()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def bos_token_id(self) -> int:
        """Returns the ID of the beginning-of-sequence token."""
        return getattr(self.tokenizer, "start_token_id", 2)

    @property
    def eos_token_id(self) -> int:
        """Returns the ID of the end-of-sequence token."""
        return getattr(self.tokenizer, "end_token_id", 1)

    @property
    def pad_token_id(self) -> int:
        """Returns the ID of the padding token."""
        return getattr(self.tokenizer, "pad_token_id", 0)

    @property
    def all_special_ids(self) -> List[int]:
        """Returns a list of all special token IDs."""
        special_ids = [self.bos_token_id, self.eos_token_id, self.pad_token_id]
        return [token_id for token_id in special_ids if token_id is not None]

    @property
    def all_special_tokens(self) -> List[str]:
        """Returns a list of all special tokens."""
        return ["<bos>", "<eos>", "<pad>"]

    def get_vocab(self) -> Dict[str, int]:
        """Returns a dictionary mapping tokens to integer IDs.
        
        vLLM expects this to construct its tokenizer cache keys.
        
        Returns:
            A dictionary mapping strings to their token IDs.
        """
        if hasattr(self.tokenizer, "get_vocabulary"):
            vocab_list = self.tokenizer.get_vocabulary()
            return {token: idx for idx, token in enumerate(vocab_list)}
        
        # Fallback if get_vocabulary is not exposed
        vocab = {}
        for i in range(self.vocab_size):
            try:
                vocab[self.tokenizer.id_to_token(i)] = i
            except Exception:
                vocab[str(i)] = i
        return vocab

    def encode(self, text: str, **kwargs: Any) -> List[int]:
        """Converts text to token IDs using the native Keras Hub tokenizer.
        
        Args:
            text: The text to encode.
            **kwargs: Additional keyword arguments.
            
        Returns:
            A list of token IDs.
        """
        from keras import ops
        token_ids = self.tokenizer(text)
        return ops.convert_to_numpy(token_ids).tolist()

    def decode(self, token_ids: List[int], **kwargs: Any) -> str:
        """Converts token IDs back to text using the native Keras Hub tokenizer.
        
        Args:
            token_ids: A list of token IDs.
            **kwargs: Additional keyword arguments.
            
        Returns:
            The decoded string.
        """
        from keras import ops
        text = self.tokenizer.detokenize(token_ids)
        try:
            np_text = ops.convert_to_numpy(text)
            if hasattr(np_text, "item"):
                np_text = np_text.item()
            if isinstance(np_text, bytes):
                return np_text.decode("utf-8")
            return str(np_text)
        except Exception:
            return str(text)
