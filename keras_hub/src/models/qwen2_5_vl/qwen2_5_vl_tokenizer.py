import keras
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


# ── All special tokens from the Qwen2.5-VL vocabulary ─────────────
# Passed to BytePairTokenizer so they are matched atomically before BPE runs.
# Without this, <|vision_start|> would be split into byte pieces instead of
# encoding to a single token id (151652).

QWEN2_5_VL_SPECIAL_TOKENS = [
    "<|endoftext|>",        # 151643  pad
    "<|im_start|>",         # 151644  chat turn start
    "<|im_end|>",           # 151645  chat turn end / eos
    "<|object_ref_start|>", # 151646
    "<|object_ref_end|>",   # 151647
    "<|box_start|>",        # 151648
    "<|box_end|>",          # 151649
    "<|quad_start|>",       # 151650
    "<|quad_end|>",         # 151651
    "<|vision_start|>",     # 151652
    "<|vision_end|>",       # 151653
    "<|vision_pad|>",       # 151654
    "<|image_pad|>",        # 151655
    "<|video_pad|>",        # 151656
    "<tool_call>",          # 151657
    "</tool_call>",         # 151658
    "<|fim_prefix|>",       # 151659
    "<|fim_middle|>",       # 151660
    "<|fim_suffix|>",       # 151661
    "<|fim_pad|>",          # 151662
    "<|repo_name|>",        # 151663
    "<|file_sep|>",         # 151664
]


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLTokenizer(BytePairTokenizer):
    """
    Qwen2.5-VL tokenizer based on byte-level BPE.

    Follows the Qwen2Tokenizer design exactly:
      - GPT-2 style regex pre-tokenization
      - Byte-level BPE encoding
      - No BOS token
      - EOS = <|im_end|>  (id 151645)
      - PAD = <|endoftext|> (id 151643)
      - Full set of multimodal special tokens

    Special token IDs (fixed by the Qwen2.5-VL vocabulary):
    ┌─────────────────────┬────────┐
    │ Token               │   ID   │
    ├─────────────────────┼────────┤
    │ <|endoftext|>       │ 151643 │
    │ <|im_start|>        │ 151644 │
    │ <|im_end|>          │ 151645 │
    │ <|object_ref_start|>│ 151646 │
    │ <|object_ref_end|>  │ 151647 │
    │ <|box_start|>       │ 151648 │
    │ <|box_end|>         │ 151649 │
    │ <|quad_start|>      │ 151650 │
    │ <|quad_end|>        │ 151651 │
    │ <|vision_start|>    │ 151652 │
    │ <|vision_end|>      │ 151653 │
    │ <|vision_pad|>      │ 151654 │
    │ <|image_pad|>       │ 151655 │
    │ <|video_pad|>       │ 151656 │
    └─────────────────────┴────────┘

    Parameters
    ----------
    vocabulary : str or dict
        Path to a `tokenizer.json` file or a pre-loaded vocabulary dict.
    merges : str or list
        Path to a `merges.txt` file or a pre-loaded merges list.

    Example
    -------
    tokenizer = Qwen2_5_VLTokenizer(
        vocabulary="tokenizer.json",
        merges="merges.txt",
    )
    token_ids = tokenizer("Hello, world!")
    text = tokenizer.detokenize(token_ids)
    """

    # ── Special token string constants ──────────────────────
    end_of_text_token    = "<|endoftext|>"
    im_start_token       = "<|im_start|>"
    im_end_token         = "<|im_end|>"
    object_ref_start     = "<|object_ref_start|>"
    object_ref_end       = "<|object_ref_end|>"
    box_start_token      = "<|box_start|>"
    box_end_token        = "<|box_end|>"
    quad_start_token     = "<|quad_start|>"
    quad_end_token       = "<|quad_end|>"
    vision_start_token   = "<|vision_start|>"
    vision_end_token     = "<|vision_end|>"
    vision_pad_token     = "<|vision_pad|>"
    image_pad_token      = "<|image_pad|>"
    video_pad_token      = "<|video_pad|>"

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
        # Build a regex that matches any special token as a literal string.
        # Sorted longest-first to avoid partial matches.
        # Used in tokenize() to split input before BPE runs so special
        # tokens are never broken into byte pieces by BPE.
        import re
        escaped = sorted(
            [re.escape(t) for t in QWEN2_5_VL_SPECIAL_TOKENS],
            key=len,
            reverse=True,
        )
        self._special_token_re = re.compile("(" + "|".join(escaped) + ")")

    def tokenize(self, inputs):
        """
        Tokenize inputs with special tokens treated as atomic units.

        This method is designed to handle both single strings and batches
        of strings. For single strings, it returns a 1D tensor of token IDs.
        For batches, it tokenizes each string and pads them to a uniform length,
        returning a 2D tensor of token IDs.
        """
        import numpy as np
        import keras

        # Check if the input is a batch (list, tuple, or a non-scalar array/tensor)
        is_batch = (
            isinstance(inputs, (list, tuple))
            or (isinstance(inputs, (np.ndarray, keras.KerasTensor)) and inputs.ndim > 0)
        )

        if not is_batch:
            # Handle single scalar input (str, bytes, scalar tensor/ndarray)
            if isinstance(inputs, bytes):
                text = inputs.decode("utf-8")
            elif isinstance(inputs, str):
                text = inputs
            else: # Must be scalar tensor or ndarray
                raw  = np.asarray(inputs).item()
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            texts_to_process = [text]
            return_scalar = True
        else:
            # Handle batch input
            if isinstance(inputs, (np.ndarray, keras.KerasTensor)):
                texts_to_process = [
                    t.decode("utf-8") if isinstance(t, bytes) else str(t)
                    for t in np.asarray(inputs).tolist()
                ]
            else: # Python list/tuple of strings
                texts_to_process = [
                    t.decode("utf-8") if isinstance(t, bytes) else str(t)
                    for t in inputs
                ]
            return_scalar = False

        all_ids = []
        for text_item in texts_to_process:
            # Split on special token boundaries, encode each segment
            parts     = self._special_token_re.split(text_item)
            token_ids = []
            for part in parts:
                if not part:
                    continue
                if self._special_token_re.fullmatch(part):
                    # Special token ─ direct vocab lookup, bypass BPE
                    token_ids.append(self.token_to_id(part))
                else:
                    # Regular text ─ delegate to parent BPE tokenizer
                    # The parent's tokenize should handle single string and return 1D tensor
                    ids = super().tokenize(part)
                    token_ids.extend(ids.numpy().tolist())
            all_ids.append(token_ids)

        if return_scalar:
            return keras.ops.convert_to_tensor(all_ids[0], dtype="int32")
        else:
            # Batch: pad to uniform length
            max_len = max(len(ids) for ids in all_ids) if all_ids else 0
            pad_id  = self.pad_token_id
            padded  = [
                ids + [pad_id] * (max_len - len(ids))
                for ids in all_ids
            ]
            return keras.ops.convert_to_tensor(padded, dtype="int32")

    # ── Special token ID properties ────────────────────────

    @property
    def pad_token_id(self):
        """<|endoftext|> is used as the padding token (id 151643)."""
        return self.token_to_id(self.end_of_text_token)

    @property
    def eos_token_id(self):
        """<|im_end|> is the end-of-sequence token (id 151645)."""
        return self.token_to_id(self.im_end_token)

    @property
    def im_start_id(self):
        return self.token_to_id(self.im_start_token)

    @property
    def im_end_id(self):
        return self.token_to_id(self.im_end_token)

    @property
    def vision_start_id(self):
        return self.token_to_id(self.vision_start_token)

    @property
    def vision_end_id(self):
        return self.token_to_id(self.vision_end_token)

    @property
    def vision_pad_id(self):
        return self.token_to_id(self.vision_pad_token)

    @property
    def image_pad_id(self):
        return self.token_to_id(self.image_pad_token)

    @property
    def video_pad_id(self):
        return self.token_to_id(self.video_pad_token)

    @property
    def object_ref_start_id(self):
        return self.token_to_id(self.object_ref_start)

    @property
    def object_ref_end_id(self):
        return self.token_to_id(self.object_ref_end)

    @property
    def box_start_id(self):
        return self.token_to_id(self.box_start_token)

    @property
    def box_end_id(self):
        return self.token_to_id(self.box_end_token)

    @property
    def quad_start_id(self):
        return self.token_to_id(self.quad_start_token)

    @property
    def quad_end_id(self):
        return self.token_to_id(self.quad_end_token)

    # ── Chat template helper ─────────────────────────────

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """
        Apply the Qwen2.5-VL chat template to a list of messages.

        Follows the official template exactly:
          - Inserts a default system prompt if the first message is not system
          - Wraps each turn with <|im_start|>{role}\\n{content}<|im_end|>\\n
          - Replaces image placeholders with <|vision_start|><|image_pad|><|vision_end|>
          - Replaces video placeholders with <|vision_start|><|video_pad|><|vision_end|>
          - Appends <|im_start|>assistant\\n if add_generation_prompt=True

        Parameters
        ----------
        messages : list of dict
            Each dict has keys "role" and "content".
            Content can be a string or a list of dicts with "type" key.
            Supported content types: "text", "image", "video".
        add_generation_prompt : bool
            If True, appends the assistant turn opener.

        Returns
        -------
        str : formatted prompt string ready for tokenization.

        Example
        -------
        messages = [
            {"role": "user", "content": "Hello, who are you?"}
        ]
        prompt = tokenizer.apply_chat_template(messages)
        token_ids = tokenizer(prompt)
        """
        result = []
        image_count = 0
        video_count = 0

        # Insert default system prompt if first message is not system
        if messages and messages[0]["role"] != "system":
            result.append(
                f"{self.im_start_token}system\n"
                f"You are a helpful assistant."
                f"{self.im_end_token}\n"
            )

        for message in messages:
            role    = message["role"]
            content = message["content"]

            result.append(f"{self.im_start_token}{role}\n")

            if isinstance(content, str):
                result.append(content)
            else:
                # Multimodal content list
                for item in content:
                    item_type = item.get("type", "")

                    if item_type == "image" or "image" in item or "image_url" in item:
                        image_count += 1
                        result.append(
                            f"{self.vision_start_token}"
                            f"{self.image_pad_token}"
                            f"{self.vision_end_token}"
                        )
                    elif item_type == "video" or "video" in item:
                        video_count += 1
                        result.append(
                            f"{self.vision_start_token}"
                            f"{self.video_pad_token}"
                            f"{self.vision_end_token}"
                        )
                    elif "text" in item:
                        result.append(item["text"])

            result.append(f"{self.im_end_token}\n")

        if add_generation_prompt:
            result.append(f"{self.im_start_token}assistant\n")

        return "".join(result)

    # ── Serialization ────────────────────

    def get_config(self):
        return super().get_config()
