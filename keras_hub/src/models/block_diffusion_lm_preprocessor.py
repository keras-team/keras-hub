import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.BlockDiffusionLMPreprocessor")
class BlockDiffusionLMPreprocessor(Preprocessor):
    """Base class for diffusion language model preprocessing layers.

    `DiffusionLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
    create a preprocessing layer for discrete block-diffusion generation tasks.
    It is intended to be paired with a `DiffusionLM` task.

    All `DiffusionLMPreprocessor` layers take a single string or batch of
    strings as input.  The prompt tokens are packed with start/end tokens and
    padded to `sequence_length`.  A canvas suffix of `canvas_length` mask
    tokens is appended after the packed prompt to form the full input for the
    model's generation API.

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

    def _call(self, x, y=None, sample_weight=None, sequence_length=None):
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
        return self._call(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if in_tf_function():
            return self._call_tf(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        return self._call(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    def _generate_preprocess(self, x, sequence_length=None):
        if not self.built:
            self.build(None)

        x = self.tokenizer(x)
        token_ids, padding_mask = self.packer(
            x,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )

        # Append canvas_length mask tokens (pad_token_id) after the prompt.
        # These placeholder positions will be filled by the denoising loop.
        mask_token_id = self.tokenizer.pad_token_id
        if len(token_ids.shape) == 1:
            canvas_shape = (self.canvas_length,)
            concat_axis = 0
        else:
            batch_size = keras.ops.shape(token_ids)[0]
            canvas_shape = (batch_size, self.canvas_length)
            concat_axis = 1
        canvas_tokens = keras.ops.full(
            canvas_shape,
            fill_value=mask_token_id,
            dtype=token_ids.dtype,
        )
        canvas_mask = keras.ops.zeros(canvas_shape, dtype=padding_mask.dtype)
        token_ids = keras.ops.concatenate(
            [token_ids, canvas_tokens], axis=concat_axis
        )
        padding_mask = keras.ops.concatenate(
            [padding_mask, canvas_mask], axis=concat_axis
        )
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        return self._generate_preprocess(x, sequence_length=sequence_length)

    def generate_preprocess(self, x, sequence_length=None):
        """Convert strings to integer token input for generation.

        Tokenizes and packs the prompt, then appends `canvas_length` mask
        tokens as placeholder canvas positions for the denoising loop.

        Args:
            x: string or batch of strings.
            sequence_length: optional int. Prompt sequence length. Defaults
                to `self.sequence_length`.

        Returns:
            A dict with keys `"token_ids"` and `"padding_mask"`.
        """
        if in_tf_function():
            return self._generate_preprocess_tf(
                x, sequence_length=sequence_length
            )
        return self._generate_preprocess(x, sequence_length=sequence_length)

    def _generate_postprocess(self, x):
        if not self.built:
            self.build(None)

        # x contains the denoised canvas token ids, shape (B, canvas_length).
        # Strip padding / special tokens and detokenize to strings.
        token_ids = keras.ops.convert_to_numpy(x).astype("int32")
        ids_to_strip = getattr(self.tokenizer, "special_token_ids", [])

        def _strip_and_detokenize(ids):
            mask = [True] * len(ids)
            for sid in ids_to_strip:
                mask = [m and (t != sid) for m, t in zip(mask, ids)]
            filtered = [t for t, m in zip(ids, mask) if m]
            return filtered

        if token_ids.ndim == 1:
            token_ids = _strip_and_detokenize(token_ids.tolist())
            return self.tokenizer.detokenize([token_ids])[0]
        result = [_strip_and_detokenize(row.tolist()) for row in token_ids]
        return self.tokenizer.detokenize(result)

    @preprocessing_function
    def _generate_postprocess_tf(self, x):
        return self._generate_postprocess(x)

    def generate_postprocess(self, x):
        """Convert denoised integer tokens back to strings.

        Args:
            x: int tensor of shape `(B, canvas_length)` produced by the
                denoising loop.

        Returns:
            String or list of strings.
        """
        if in_tf_function():
            return self._generate_postprocess_tf(x)
        return self._generate_postprocess(x)

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

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            config["tokenizer"] = keras.saving.deserialize_keras_object(
                config["tokenizer"]
            )
        return cls(**config)
