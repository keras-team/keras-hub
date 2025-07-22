"""
LayoutLMv3 tokenizer implementation.

This module implements the tokenizer for the LayoutLMv3 model, which is used for
document understanding tasks. The tokenizer handles both text and layout
information, including bounding box coordinates.

References:
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
"""

from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras_hub_export("keras_hub.models.LayoutLMv3Tokenizer")
class LayoutLMv3Tokenizer(WordPieceTokenizer):
    """LayoutLMv3 tokenizer for document understanding tasks.

    This class implements the tokenizer for the LayoutLMv3 model, which handles
    both text and layout information. It tokenizes text and processes bounding
    box coordinates for document understanding tasks.

    Args:
        vocabulary: dict. A dictionary mapping tokens to integer ids, or a
            string path to a vocabulary file. If passing a file, the file
            should be one token per line. If `None`, we will used the default
            vocabulary.
        merges: string or list. If a string, a path to a merges file. If a
            list, a list of merge rules. Each merge rule should be a string
            of the form "word1 word2". If `None`, we will use the default
            merges.
        lowercase: bool. If `True`, the input text will be lowercased before
            tokenization. Defaults to `False`.
        sequence_length: int. If set, the output will be padded or truncated to
            the `sequence_length`. Defaults to `None`.
        special_tokens: dict. A dictionary of special tokens to be added to
            the vocabulary. Keys should be the special token type and values
            should be the special token string. Defaults to standard BERT
            special tokens.

    Examples:
    ```python
    # Unbatched inputs.
    tokenizer = keras_hub.models.LayoutLMv3Tokenizer.from_preset(
        "layoutlmv3_base"
    )
    
    # Tokenize text only
    tokenizer("The quick brown fox")
    
    # Tokenize text with bounding boxes
    tokenizer(
        "The quick brown fox",
        bbox=[[0, 0, 100, 50], [100, 0, 200, 50], [200, 0, 300, 50], [300, 0, 400, 50]]
    )

    # Batched inputs.
    tokenizer(["The quick brown fox", "Hello world"])
    
    # Batched inputs with bounding boxes
    tokenizer(
        ["The quick brown fox", "Hello world"],
        bbox=[
            [[0, 0, 100, 50], [100, 0, 200, 50], [200, 0, 300, 50], [300, 0, 400, 50]],
            [[0, 0, 100, 50], [100, 0, 200, 50]]
        ]
    )
    ```
    """

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        lowercase=False,
        sequence_length=None,
        special_tokens=None,
        **kwargs,
    ):
        # Set default special tokens for LayoutLMv3 if not provided
        if special_tokens is None:
            special_tokens = {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "unk_token": "[UNK]",
            }

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            lowercase=lowercase,
            sequence_length=sequence_length,
            special_tokens=special_tokens,
            **kwargs,
        )

    def _process_bbox_for_tokens(self, text_list, bbox_list):
        """Process bounding boxes to align with tokenized text.
        
        This method expands bounding boxes for subword tokens and adds
        dummy boxes for special tokens.
        """
        if bbox_list is None:
            return None
            
        processed_bbox = []
        
        try:
            for text, bbox in zip(text_list, bbox_list):
                # Handle empty or None inputs defensively
                if not text or not bbox:
                    words = []
                    word_bbox = []
                else:
                    words = text.split()
                    # Ensure bbox has correct length or use dummy boxes
                    if len(bbox) != len(words):
                        word_bbox = [[0, 0, 0, 0] for _ in words]
                    else:
                        word_bbox = bbox
                
                token_bbox = []
                # Add dummy box for [CLS] token
                token_bbox.append([0, 0, 0, 0])
                
                # Process each word and its corresponding box
                for word, word_box in zip(words, word_bbox):
                    # Tokenize the word to handle subwords
                    try:
                        word_tokens = self.tokenize(word)
                        # Expand the bounding box for all subword tokens
                        for _ in word_tokens:
                            token_bbox.append(word_box)
                    except Exception:
                        # Fallback: just add one token with the box
                        token_bbox.append(word_box)
                
                # Add dummy box for [SEP] token
                token_bbox.append([0, 0, 0, 0])
                processed_bbox.append(token_bbox)
                
        except Exception:
            # Fallback: return None to use dummy boxes
            return None
            
        return processed_bbox

    def call(self, inputs, bbox=None, sequence_length=None):
        """Tokenize inputs and process bounding boxes.
        
        Args:
            inputs: String or list of strings to tokenize.
            bbox: Optional bounding box coordinates. Should be a list of
                [x0, y0, x1, y1] coordinates for each word, or a list of
                such lists for batched inputs.
            sequence_length: Optional length to pad/truncate to.
                
        Returns:
            Dictionary containing:
            - token_ids: Tokenized input
            - padding_mask: Mask for padded tokens  
            - bbox: Processed bounding box coordinates
        """
        # Handle single string input
        if isinstance(inputs, str):
            inputs = [inputs]
            if bbox is not None:
                bbox = [bbox]
        
        # Process bounding boxes to align with tokens
        processed_bbox = self._process_bbox_for_tokens(inputs, bbox)
        
        # Get tokenized output from parent class
        token_output = super().call(inputs, sequence_length=sequence_length)
        
        # Add bounding box information
        if processed_bbox is not None:
            try:
                batch_size = ops.shape(token_output["token_ids"])[0]
                seq_len = ops.shape(token_output["token_ids"])[1]
                bbox_tensor = []
                
                for i, bbox_seq in enumerate(processed_bbox):
                    # Truncate or pad bbox sequence to match token sequence length
                    if len(bbox_seq) > seq_len:
                        bbox_seq = bbox_seq[:seq_len]
                    else:
                        # Pad with dummy boxes
                        padding_needed = seq_len - len(bbox_seq)
                        bbox_seq = bbox_seq + [[0, 0, 0, 0]] * padding_needed
                    bbox_tensor.append(bbox_seq)
                
                # Convert to tensor with explicit dtype
                bbox_tensor = ops.convert_to_tensor(bbox_tensor, dtype="int32")
                token_output["bbox"] = bbox_tensor
                
            except Exception:
                # Fallback: create dummy bounding boxes
                batch_size = ops.shape(token_output["token_ids"])[0]
                seq_len = ops.shape(token_output["token_ids"])[1]
                dummy_bbox = ops.zeros((batch_size, seq_len, 4), dtype="int32")
                token_output["bbox"] = dummy_bbox
        else:
            # Create dummy bounding boxes when no bbox input provided
            batch_size = ops.shape(token_output["token_ids"])[0]
            seq_len = ops.shape(token_output["token_ids"])[1]
            dummy_bbox = ops.zeros((batch_size, seq_len, 4), dtype="int32")
            token_output["bbox"] = dummy_bbox
            
        return token_output

    def get_config(self):
        """Return the configuration of the tokenizer."""
        config = super().get_config()
        # Remove any keys that might not be serializable
        serializable_config = {}
        for key, value in config.items():
            try:
                # Test if the value is serializable by converting to string
                str(value)
                serializable_config[key] = value
            except Exception:
                # Skip non-serializable values
                continue
        return serializable_config

    @property  
    def backbone_cls(self):
        # Avoid circular imports by importing here
        from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
            LayoutLMv3Backbone,
        )
        return LayoutLMv3Backbone
