"""
LayoutLMv3 tokenizer implementation.

This module implements the tokenizer for the LayoutLMv3 model, which is used for
document understanding tasks. The tokenizer handles both text and layout
information, including bounding box coordinates.

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

    This class implements the tokenizer for the LayoutLMv3 model, which handles
    both text and layout information. It tokenizes text and processes bounding
    box coordinates for document understanding tasks.

    Args:
        vocabulary: dict. A dictionary mapping tokens to integer ids, or a
            string path to a vocabulary file. If passing a file, the file
            should be one token per line. If `None`, we will used the default
            vocabulary for the given model preset.
        lowercase: bool. If `True`, the input text will be lowercased before
            tokenization. Defaults to `True`.
        strip_accents: bool. If `True`, all accent marks will be removed from
            text before tokenization. Defaults to `None` (no stripping).
        split: bool. If `True`, input will be split on whitespace before
            tokenization. Defaults to `True`.
        split_on_cjk: bool. If `True`, input will be split on CJK characters
            before tokenization. CJK characters include Chinese, Japanese, and
            Korean. Defaults to `True`.
        suffix_indicator: str. The characters prepended to a wordpiece to
            indicate that it is a suffix to another subword. E.g. "##" for BERT.
            Defaults to `"##"`.
        oov_token: str. The out of vocabulary token to use when a word cannot
            be found in the vocabulary. Defaults to `"[UNK]"`.
        **kwargs: additional keyword arguments to pass to the parent class.

    Examples:
    ```python
    # Tokenize a simple string.
    tokenizer = keras_hub.models.LayoutLMv3Tokenizer.from_preset(
        "layoutlmv3_base",
    )
    tokenizer("The quick brown fox.")

    # Tokenize a list of strings.
    tokenizer(["The quick brown fox.", "The fox trots."])

    # Tokenize text with bounding boxes.
    tokenizer(
        ["Hello world"],
        bbox=[[[0, 0, 100, 50], [100, 0, 200, 50]]]
    )

    # Custom vocabulary.
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=10,
        model_type="WORD",
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    tokenizer = keras_hub.models.LayoutLMv3Tokenizer(
        vocabulary=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox.")
    ```
    """

    def __init__(
        self,
        vocabulary=None,
        lowercase=True,
        strip_accents=None,
        split=True,
        split_on_cjk=True,
        suffix_indicator="##",
        oov_token="[UNK]",
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

        # Special tokens
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"

    def _process_bbox_for_tokens(self, text_list, bbox_list):
        """Process bounding boxes to align with tokenized text.
        
        This method handles the expansion of bounding boxes to match subword
        tokenization and adds dummy bounding boxes for special tokens.
        
        Args:
            text_list: List of strings to tokenize.
            bbox_list: List of lists of bounding boxes corresponding to words.
            
        Returns:
            Processed bounding boxes aligned with tokens.
        """
        if bbox_list is None:
            return None
            
        processed_bbox = []
        
        for text, bbox in zip(text_list, bbox_list):
            # Split text into words for alignment
            words = text.split()
            
            # Ensure bbox list matches word count
            if len(bbox) != len(words):
                # If bbox count doesn't match word count, use dummy boxes
                word_bbox = [[0, 0, 0, 0] for _ in words]
            else:
                word_bbox = bbox
            
            # Tokenize each word to see how many tokens it becomes
            token_bbox = []
            
            # Add dummy bbox for [CLS] token
            token_bbox.append([0, 0, 0, 0])
            
            for word, word_box in zip(words, word_bbox):
                # Get tokens for this word
                word_tokens = self.tokenize(word)
                
                # Add the same bounding box for all tokens of this word
                for _ in word_tokens:
                    token_bbox.append(word_box)
            
            # Add dummy bbox for [SEP] token
            token_bbox.append([0, 0, 0, 0])
            
            processed_bbox.append(token_bbox)
            
        return processed_bbox

    def call(self, inputs, bbox=None, sequence_length=None):
        """Tokenize strings and optionally pack sequences.

        Args:
            inputs: A string, list of strings, or dict of string tensors.
            bbox: Optional list of bounding box coordinates for each input text.
                Should be a list of lists of [x0, y0, x1, y1] coordinates
                corresponding to words in the input text.
            sequence_length: int. If set, the output will be packed or padded
                to exactly this sequence length.

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
