import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


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

    @preprocessing_function
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
            raise ValueError("sequence_length must be specified.")
        if keras.config.backend() in ["torch", "jax"]:
            # When using rwkv_ops, ensure sequence_length is divisible by 16.
            try:
                import rwkv_ops  # noqa: F401

                if sequence_length % 16 != 0:
                    sequence_length += (16 - sequence_length % 16) % 16
            except ImportError:
                pass
        x = self.tokenizer(x)

        token_ids, padding_mask = self.packer(
            x, sequence_length=sequence_length + 1, add_end_value=False
        )

        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
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

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
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

        x = [t[-sequence_length:] for t in self.tokenizer(x)]
        y = tf.zeros((len(x), generate_length), "int32")
        # Utilize RNN characteristics where prefill and decode are two sequences
        # But the first token of decode should be the last token of prefill
        start_token = [[t[-1]] for t in x]
        x = [np.array(t[:-1]) if len(t) > 1 else [0] for t in x]
        x = tf.ragged.constant(x)
        token_ids, input_padding_mask = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )
        start_token = tf.convert_to_tensor(start_token, "int32")

        y = tf.concat([start_token, y], axis=1)
        padding_mask = tf.not_equal(y, 0)

        return {
            "token_ids": token_ids,
            "input_padding_mask": input_padding_mask,
            "padding_mask": padding_mask,
            "predict_token_ids": y,
        }

    @preprocessing_function
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
        return self.tokenizer.detokenize(token_ids * padding_mask)
