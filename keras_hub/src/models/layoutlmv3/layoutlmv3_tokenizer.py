"""
LayoutLMv3 tokenizer for document understanding tasks.

References:
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
"""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras_hub_export("keras_hub.models.LayoutLMv3Tokenizer")
class LayoutLMv3Tokenizer(WordPieceTokenizer):
    """LayoutLMv3 tokenizer for document understanding tasks.

    This tokenizer is specifically designed for LayoutLMv3 models that process
    both text and layout information. It tokenizes text and processes bounding
    box coordinates for document understanding tasks.

    Args:
        vocabulary: Optional list of strings containing the vocabulary. If None,
            vocabulary will be loaded from preset.
        lowercase: bool, defaults to True. Whether to lowercase the input text.
        strip_accents: bool, defaults to True. Whether to strip accents from
            the input text.
        split: bool, defaults to True. Whether to split the input on whitespace.
        split_on_cjk: bool, defaults to True. Whether to split CJK characters.
        suffix_indicator: str, defaults to "##". The prefix to add to 
            continuation tokens.
        oov_token: str, defaults to "[UNK]". The out-of-vocabulary token.
        cls_token: str, defaults to "[CLS]". The classification token.
        sep_token: str, defaults to "[SEP]". The separator token.
        pad_token: str, defaults to "[PAD]". The padding token.
        mask_token: str, defaults to "[MASK]". The mask token.
        unk_token: str, defaults to "[UNK]". The unknown token.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
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
    """

    def __init__(
        self,
        vocabulary=None,
        lowercase=True,
        strip_accents=True,
        split=True,
        split_on_cjk=True,
        suffix_indicator="##",
        oov_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            strip_accents=strip_accents,
            split=split,
            split_on_cjk=split_on_cjk,
            suffix_indicator=suffix_indicator,
            oov_token=oov_token,
            **kwargs,
        )
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token

    def _process_bbox_for_tokens(self, text_list, bbox_list):
        """This method expands bounding boxes for subword tokens and adds
        dummy boxes for special tokens.
        
        Args:
            text_list: List of text strings.
            bbox_list: List of bounding box lists corresponding to words.
            
        Returns:
            List of bounding box lists aligned with tokens, or None if bbox_list is None.
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
        """Tokenize input text and process bounding boxes.

        Args:
            inputs: A string, list of strings, or tensor of strings to tokenize.
            bbox: Optional bounding box coordinates corresponding to the words
                in the input text. Should be a list of lists of [x0, y0, x1, y1]
                coordinates for each word.
            sequence_length: int. If set, the output will be packed or padded to
                exactly this sequence length.

        Returns:
            A dictionary with the tokenized inputs and optionally bounding boxes.
            If input is a string or list of strings, the dictionary will contain:
            - "token_ids": Tokenized representation of the inputs.
            - "padding_mask": A mask indicating which tokens are real vs padding.
            - "bbox": Bounding box coordinates aligned with tokens (if provided).
        """
        # Handle string inputs by converting to list
        if isinstance(inputs, str):
            inputs = [inputs]
            if bbox is not None:
                bbox = [bbox]

        # Process bounding boxes before tokenization
        processed_bbox = self._process_bbox_for_tokens(inputs, bbox)

        # Tokenize the text
        token_output = super().call(inputs, sequence_length=sequence_length)
        
        # Process bbox if provided
        if processed_bbox is not None:
            # Convert to tensors and pad to match token sequence length
            batch_size = ops.shape(token_output["token_ids"])[0]
            seq_len = ops.shape(token_output["token_ids"])[1]
            
            # Create bbox tensor
            bbox_tensor = []
            for i, bbox_seq in enumerate(processed_bbox):
                # Pad or truncate bbox sequence to match token sequence
                if len(bbox_seq) > seq_len:
                    bbox_seq = bbox_seq[:seq_len]
                else:
                    # Pad with dummy boxes
                    bbox_seq = bbox_seq + [[0, 0, 0, 0]] * (seq_len - len(bbox_seq))
                bbox_tensor.append(bbox_seq)
            
            # Convert to tensor
            bbox_tensor = ops.convert_to_tensor(bbox_tensor, dtype="int32")
            token_output["bbox"] = bbox_tensor
        else:
            # Create dummy bbox tensor if no bbox provided
            batch_size = ops.shape(token_output["token_ids"])[0]
            seq_len = ops.shape(token_output["token_ids"])[1]
            dummy_bbox = ops.zeros((batch_size, seq_len, 4), dtype="int32")
            token_output["bbox"] = dummy_bbox

        return token_output

    def get_config(self):
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
