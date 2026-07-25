import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged
from keras_hub.src.utils.tensor_utils import strip_to_ragged_python


@keras_hub_export("keras_hub.models.BlockDiffusionLMPreprocessor")
class BlockDiffusionLMPreprocessor(Preprocessor):
    """Base class for diffusion language model preprocessing layers.

    `BlockDiffusionLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer`
    to create a preprocessing layer for discrete block-diffusion generation
    tasks. It is intended to be paired with a `DiffusionLM` task.

    All `BlockDiffusionLMPreprocessor` layers take a single string or batch of
    strings as input.  The prompt tokens are packed with start/end tokens and
    padded to `sequence_length`.  The canvas is initialised separately inside
    `generate_step` via `_init_canvas` so that `_encode_prompt` always
    receives only the real prompt tokens.

    Subclasses should override `generate_preprocess` and `generate_postprocess`
    to handle model-specific details (e.g. multimodal inputs, special canvas
    token handling).

    Args:
        tokenizer: A `keras_hub.tokenizers.Tokenizer` instance.
        sequence_length: int. Maximum total sequence length (prompt +
            canvas). Defaults to `256`.
        canvas_length: int. Number of canvas (mask) tokens appended after
            the prompt during generation preprocessing. Defaults to `256`.
        add_start_token: bool. Whether to prepend the start token to the
            prompt. Defaults to `True`.
        add_end_token: bool. Whether to append the end token to the prompt.
            Defaults to `True`.
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=256,
        canvas_length=256,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            _allow_python_workflow=_allow_python_workflow, **kwargs
        )
        self.tokenizer = tokenizer
        self.packer = None
        self.sequence_length = sequence_length
        self.canvas_length = canvas_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token

    def build(self, input_shape):
        # Defer packer creation to `build()` so that tokenizer assets are
        # loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length
        x = self.tokenizer(x)
        token_ids, padding_mask = self.packer(
            x,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }
        y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        return self._call_python(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self._allow_python_workflow or in_tf_function():
            return self._call_tf(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        else:
            return self._call_python(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )

    def _generate_preprocess_python(self, x, sequence_length=None):
        if not self.built:
            self.build(None)

        x = self.tokenizer(x)
        token_ids, padding_mask = self.packer(
            x,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        return self._generate_preprocess_python(
            x, sequence_length=sequence_length
        )

    def generate_preprocess(self, x, sequence_length=None):
        """Convert strings to integer token input for generation.

        Tokenizes and packs the prompt.  The canvas is initialised inside
        `generate_step` via `_init_canvas`; no canvas tokens are appended.

        Args:
            x: string or batch of strings.
            sequence_length: optional int. Prompt sequence length. Defaults
                to `self.sequence_length`.

        Returns:
            A dict with keys `"token_ids"` and `"padding_mask"`.
        """
        if not self._allow_python_workflow or in_tf_function():
            return self._generate_preprocess_tf(
                x, sequence_length=sequence_length
            )
        else:
            return self._generate_preprocess_python(
                x, sequence_length=sequence_length
            )

    def _generate_postprocess_python(self, x):
        if not self.built:
            self.build(None)
        ids_to_strip = getattr(self.tokenizer, "special_token_ids", [])
        was_1d = keras.ops.ndim(x) == 1
        # All canvas positions are valid (no padding); mask=all-True strips
        # only special tokens.
        mask = keras.ops.ones_like(x, dtype="bool")
        token_ids = strip_to_ragged_python(x, mask, ids_to_strip)
        if was_1d:
            return self.tokenizer.detokenize([token_ids])[0]
        return self.tokenizer.detokenize(token_ids)

    @preprocessing_function
    def _generate_postprocess_tf(self, x):
        if not self.built:
            self.build(None)
        ids_to_strip = self.tokenizer.special_token_ids
        mask = keras.ops.ones_like(x, dtype="bool")
        token_ids = strip_to_ragged(x, mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

    def generate_postprocess(self, x):
        """Convert denoised integer tokens back to strings.

        Args:
            x: int tensor of shape `(B, canvas_length)` produced by the
                denoising loop.

        Returns:
            String or list of strings.
        """
        if not self._allow_python_workflow or in_tf_function():
            return self._generate_postprocess_tf(x)
        else:
            return self._generate_postprocess_python(x)

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        if self.packer is not None:
            self.packer.sequence_length = value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "canvas_length": self.canvas_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
            }
        )
        return config
