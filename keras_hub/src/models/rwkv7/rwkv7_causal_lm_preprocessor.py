import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer


@keras_hub_export("keras_hub.models.RWKV7CausalLMPreprocessor")
class RWKV7CausalLMPreprocessor(CausalLMPreprocessor):
    """RWKV-7 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.RWKV7CausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.RWKV7CausalLM` instance, these methods
    will be called implicitly in generate(). They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.RWKVTokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Default is `False`.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured sequence_length of
            the layer.


    Examples:
    ```python
    # Initialize the tokenizer and load assets from a local path.
    tokenizer = RWKVTokenizer()
    tokenizer.load_assets(rwkv_path)

    # Create a preprocessor with a sequence length of 8.
    preprocessor = RWKV7CausalLMPreprocessor(tokenizer, sequence_length=8)

    # Tokenize and pack a batch of sentences.
    preprocessor(["Bubble sort\n```python", "Hello World\n```python\n"])

    # Preprocess inputs for generation with a maximum generation length of 16.
    preprocessor.generate_preprocess(
        ["Bubble sort\n```python", "Hello World\n```python\n"], 16
    )
    ```
    Outputs (torch Backend) :
    tensor([[    0,  0,  0,  0,  0,  0,  0,  0,  0,   893,
          1760,  2011, 32082,    11,  6884],
            [    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            33155, 37576,    11,  6884, 42114]], dtype=torch.int32),
    tensor([[    0,  0,  0,  0,  0,  0,  0,  0,   893,  1760,
            2011, 32082,    11,  6884, 42114],
            [    0,  0,  0,  0,  0,  0,  0,  0,  0, 33155,
            37576,    11,  6884, 42114,    11]], dtype=torch.int32),
    tensor([[False, False, False, False, False, False, False, False,  True,
            True,  True,  True,  True,  True,  True],
            [False, False, False, False, False, False, False, False, False,
            True,  True,  True,  True,  True,  True]])

    {'token_ids': tensor([[    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            893,  1760,  2011, 32082,    11,  6884],
            [    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0, 33155, 37576,    11,  6884, 42114]], dtype=torch.int32),
    'padding_mask': tensor([[ True, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False]]),
    'predict_token_ids': tensor([[42114,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0],
            [   11,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0]], dtype=torch.int32)}
    """

    backbone_cls = RWKV7Backbone
    tokenizer_cls = RWKVTokenizer

    def __init__(
        self,
        tokenizer,
        add_start_token=False,
        **kwargs,
    ):
        """Initialize the preprocessor.

        Args:
            tokenizer: The tokenizer to use.
            add_start_token: Whether to add start token.
            **kwargs: Additional arguments.
        """
        super().__init__(
            tokenizer=tokenizer, add_start_token=add_start_token, **kwargs
        )

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        """Preprocess the input for training.

        Args:
            x: Input text data.
            y: Target data (optional).
            sample_weight: Sample weights (optional).
            sequence_length: Desired sequence length.

        Returns:
            Preprocessed data tuple (x, y, sample_weight).
        """
        if isinstance(x, str):
            x = [x]
        sequence_length = sequence_length or self.sequence_length
        # Pad length to multiples of 16 to meet kernel requirements
        if sequence_length is None:
            raise (ValueError("`sequence_length` must be specified."))
        if (sequence_length - 1) % 16 != 0:
            sequence_length = sequence_length + (
                16 - (sequence_length - 1) % 16
            )
        x = self.tokenizer(x)

        token_ids, padding_mask = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )

        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., 1:],
        }
        # Target `y` will be the next token.
        y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def build(self, input_shape):
        self.packer = StartEndPacker(
            start_value=None,
            end_value=None,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
            padding_side="left",  # RWKV uses left-padding exclusively
        )
        self.built = True

    def generate_preprocess(
        self,
        x,
        sequence_length,
    ):
        """Preprocess input for generation.

        Args:
            x: Input text data.
            sequence_length: Maximum generation length.

        Returns:
            Dictionary with preprocessed inputs for generation.
        """
        if isinstance(x, str):
            x = [x]

        if not self.built:
            self.build(None)
        # Align with Keras API
        # Input sequence_length is the maximum generation length
        # While self.sequence_length corresponds to the prefill max length
        generate_length = sequence_length
        if sequence_length is None:
            raise ValueError("`sequence_length` must be specified.")
        sequence_length = self.sequence_length

        # Pad length to multiples of 16 to meet kernel requirements
        if sequence_length % 16 != 0:
            sequence_length = sequence_length + (16 - sequence_length % 16)
        if generate_length % 16 != 0:
            generate_length = generate_length + (16 - generate_length % 16)

        x = [t[-sequence_length:] for t in self.tokenizer(x)]
        y = ops.zeros((len(x), generate_length), "int32")
        # Utilize RNN characteristics where prefill and decode are two sequences
        # But the first token of decode should be the last token of prefill
        start_token = [[t[-1]] for t in x]
        x = [t[:-1] if len(t) > 1 else [0] for t in x]

        token_ids, __ = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )
        start_token = ops.convert_to_tensor(start_token, "int32")
        y = ops.slice_update(y, [0, 0], start_token)
        padding_mask = ops.not_equal(y, 0)

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "predict_token_ids": y,
        }

    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.

        Args:
            x: Dictionary containing token_ids and padding_mask.

        Returns:
            Detokenized string output.
        """
        if not self.built:
            self.build(None)

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        token_ids = ops.convert_to_numpy(token_ids)
        padding_mask = ops.convert_to_numpy(padding_mask)
        return self.tokenizer.detokenize(token_ids * padding_mask)
