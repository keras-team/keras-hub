"""
LayoutLMv3 tokenizer implementation.

This module implements the tokenizer for the LayoutLMv3 model, which is used for
document understanding tasks. The tokenizer handles both text and layout
information, including bounding box coordinates.

References:
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
"""

from typing import Dict
from typing import List
from typing import Optional

from keras import backend
from keras.saving import register_keras_serializable

from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@register_keras_serializable()
class LayoutLMv3Tokenizer(WordPieceTokenizer):
    """LayoutLMv3 tokenizer for document understanding tasks.

    This class implements the tokenizer for the LayoutLMv3 model, which handles
    both text and layout information. It tokenizes text and processes bounding
    box coordinates for document understanding tasks.

    Example:
        ```python
        # Initialize tokenizer from preset
        tokenizer = LayoutLMv3Tokenizer.from_preset("layoutlmv3_base")

        # Tokenize text and bounding boxes
        inputs = tokenizer(
            text=["Hello world", "How are you"],
            bbox=[[[0, 0, 100, 100], [100, 0, 200, 100]],
                  [[0, 0, 100, 100], [100, 0, 200, 100]]]
        )
        ```

    Args:
        vocabulary: Optional list of strings containing the vocabulary. If None,
            vocabulary will be loaded from preset.
        lowercase: bool, defaults to True. Whether to lowercase the input text.
        strip_accents: bool, defaults to True. Whether to strip accents from
            the input text.
        sequence_length: int, defaults to 512. Maximum sequence length of the
            tokenized output.
        **kwargs: Additional keyword arguments passed to the parent class.

    References:
        - [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
        - [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
    """

    def __init__(
        self,
        vocabulary: Optional[List[str]] = None,
        lowercase: bool = True,
        strip_accents: bool = True,
        sequence_length: int = 512,
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            strip_accents=strip_accents,
            sequence_length=sequence_length,
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
        self.cls_token_mask = backend.constant(1, dtype="int32")
        self.sep_token_mask = backend.constant(1, dtype="int32")
        self.pad_token_mask = backend.constant(0, dtype="int32")
        self.mask_token_mask = backend.constant(1, dtype="int32")
        self.unk_token_mask = backend.constant(1, dtype="int32")

    def call(self, text, bbox=None, **kwargs):
        """Tokenize text and process bounding boxes.

        Args:
            text: A string or list of strings to tokenize.
            bbox: Optional list of bounding box coordinates for each token. If
                provided, should be a list of lists of [x0, y0, x1, y1]
                coordinates.
            **kwargs: Additional keyword arguments passed to the parent class.

        Returns:
            A dictionary containing:
                - token_ids: Tensor of shape (batch_size, sequence_length)
                  containing token IDs
                - padding_mask: Tensor of shape (batch_size, sequence_length)
                  containing padding mask
                - attention_mask: Tensor of shape (batch_size, sequence_length)
                  containing attention mask
                - bbox: Tensor of shape (batch_size, sequence_length, 4)
                  containing bounding box coordinates (if provided)
        """
        # Tokenize input text
        token_ids, padding_mask = super().call(text)

        # Add [CLS] token at the beginning
        batch_size = backend.shape(token_ids)[0]
        cls_token_ids = (
            backend.ones((batch_size, 1), dtype="int32") * self.cls_token_id
        )
        cls_token_mask = (
            backend.ones((batch_size, 1), dtype="int32") * self.cls_token_mask
        )

        token_ids = backend.concatenate([cls_token_ids, token_ids], axis=1)
        padding_mask = backend.concatenate(
            [cls_token_mask, padding_mask], axis=1
        )

        # Add [SEP] token at the end
        sep_token_ids = (
            backend.ones((batch_size, 1), dtype="int32") * self.sep_token_id
        )
        sep_token_mask = (
            backend.ones((batch_size, 1), dtype="int32") * self.sep_token_mask
        )

        token_ids = backend.concatenate([token_ids, sep_token_ids], axis=1)
        padding_mask = backend.concatenate(
            [padding_mask, sep_token_mask], axis=1
        )

        # Create attention mask
        attention_mask = backend.cast(padding_mask, dtype="int32")

        # Process bounding boxes
        if bbox is not None:
            bbox_tensor = backend.stack(bbox, axis=1)
        else:
            bbox_tensor = None

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "attention_mask": attention_mask,
            "bbox": bbox_tensor,
        }

    def detokenize(self, token_ids):
        """Convert token IDs back to text.

        Args:
            token_ids: Tensor of shape (batch_size, sequence_length) containing
                token IDs.

        Returns:
            A list of strings containing the detokenized text.
        """
        # Remove special tokens
        token_ids = token_ids[:, 1:-1]  # Remove [CLS] and [SEP]

        # Convert to text
        return super().detokenize(token_ids)

    def get_config(self) -> Dict:
        """Get the tokenizer configuration.

        Returns:
            Dictionary containing the tokenizer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
                "unk_token": self.unk_token,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict) -> "LayoutLMv3Tokenizer":
        """Create a tokenizer from a configuration dictionary.

        Args:
            config: Dictionary containing the tokenizer configuration.

        Returns:
            LayoutLMv3Tokenizer instance.
        """
        return cls(**config)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Create a LayoutLMv3 tokenizer from a preset.

        Args:
            preset: string. Must be one of "layoutlmv3_base",
                "layoutlmv3_large".
            **kwargs: Additional keyword arguments passed to the tokenizer.

        Returns:
            A LayoutLMv3Tokenizer instance.

        Raises:
            ValueError: If the preset is not supported.
        """
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}"""
            )

        metadata = cls.presets[preset]
        config = metadata["config"]
        vocabulary = metadata["vocabulary"]

        # Create tokenizer
        tokenizer = cls(
            vocabulary=vocabulary,
            sequence_length=config["sequence_length"],
            **kwargs,
        )

        return tokenizer
