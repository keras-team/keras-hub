"""
LayoutLMv3 tokenizer for document understanding tasks.

References:
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
"""

from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export("keras_hub.models.LayoutLMv3Tokenizer")
class LayoutLMv3Tokenizer(BytePairTokenizer):
    """LayoutLMv3 tokenizer for document understanding tasks.

    This tokenizer is specifically designed for LayoutLMv3 models that process
    both text and layout information. It tokenizes text and processes bounding
    box coordinates for document understanding tasks.

    Args:
        vocabulary: Optional dict or string containing the vocabulary. If None,
            vocabulary will be loaded from preset.
        merges: Optional list or string containing the merge rules for BPE.
            If None, merges will be loaded from preset.
        sequence_length: int. If set, the output will be packed or padded to
            exactly this sequence length.
        add_prefix_space: bool. Whether to add a prefix space to the input.
        unsplittable_tokens: Optional list of tokens that should never be split.
        dtype: str. The output dtype for token IDs.
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
        merges=None,
        sequence_length=None,
        add_prefix_space=False,
        unsplittable_tokens=None,
        dtype="int32",
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            sequence_length=sequence_length,
            add_prefix_space=add_prefix_space,
            unsplittable_tokens=unsplittable_tokens,
            dtype=dtype,
            **kwargs,
        )
        # Store special tokens for bbox processing
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"

    def _process_bbox_for_tokens(self, text_list, bbox_list):
        """This method expands bounding boxes for subword tokens and adds
        dummy boxes for special tokens.

        Args:
            text_list: List of text strings.
            bbox_list: List of bounding box lists corresponding to words.

        Returns:
            List of bounding box lists aligned with tokens, or None if
            bbox_list is None.
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

        except Exception as e:
            import warnings

            warnings.warn(
                f"Error processing bounding boxes: {e}. "
                f"Falling back to dummy boxes."
            )
            # Fallback: return None to use dummy boxes
            return None

        return processed_bbox

    def _apply_sequence_length(self, token_output, sequence_length):
        """Apply sequence length padding or truncation to token output."""
        token_ids = token_output["token_ids"]
        padding_mask = token_output["padding_mask"]

        # Get current sequence length
        current_seq_len = ops.shape(token_ids)[1]

        if current_seq_len > sequence_length:
            # Truncate
            token_ids = token_ids[:, :sequence_length]
            padding_mask = padding_mask[:, :sequence_length]
        elif current_seq_len < sequence_length:
            # Pad
            pad_length = sequence_length - current_seq_len
            pad_token_id = self.vocabulary.get(self.pad_token, 0)

            # Pad token_ids
            pad_tokens = ops.full(
                (ops.shape(token_ids)[0], pad_length),
                pad_token_id,
                dtype=token_ids.dtype,
            )
            token_ids = ops.concatenate([token_ids, pad_tokens], axis=1)

            # Pad padding_mask
            pad_mask = ops.zeros(
                (ops.shape(padding_mask)[0], pad_length),
                dtype=padding_mask.dtype,
            )
            padding_mask = ops.concatenate([padding_mask, pad_mask], axis=1)

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

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
            A dictionary with the tokenized inputs and optionally bounding
            boxes. If input is a string or list of strings, the dictionary
            will contain:
            - "token_ids": Tokenized representation of the inputs.
            - "padding_mask": A mask indicating which tokens are real vs
                padding.
            - "bbox": Bounding box coordinates aligned with tokens
                (if provided).
        """
        # Handle string inputs by converting to list
        if isinstance(inputs, str):
            inputs = [inputs]
            if bbox is not None:
                bbox = [bbox]

        # Process bounding boxes before tokenization
        processed_bbox = self._process_bbox_for_tokens(inputs, bbox)

        # Tokenize the text
        if sequence_length is not None:
            token_output = super().call(inputs)
            # Apply sequence length padding/truncation manually
            token_output = self._apply_sequence_length(
                token_output, sequence_length
            )
        else:
            token_output = super().call(inputs)

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
                    bbox_seq = bbox_seq + [[0, 0, 0, 0]] * (
                        seq_len - len(bbox_seq)
                    )
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

    @classmethod
    def from_config(cls, config):
        # Extract special tokens from config
        special_tokens = {
            "cls_token": config.pop("cls_token", "[CLS]"),
            "sep_token": config.pop("sep_token", "[SEP]"),
            "pad_token": config.pop("pad_token", "[PAD]"),
            "mask_token": config.pop("mask_token", "[MASK]"),
            "unk_token": config.pop("unk_token", "[UNK]"),
        }

        # Create instance using parent method
        instance = super().from_config(config)

        # Set special tokens
        for token_name, token_value in special_tokens.items():
            setattr(instance, token_name, token_value)

        return instance
