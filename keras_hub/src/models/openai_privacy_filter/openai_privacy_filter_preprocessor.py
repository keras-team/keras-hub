import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import compute_padding_mask
from keras_hub.src.utils.tensor_utils import convert_preprocessing_inputs
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import try_import_tensorflow

tf = try_import_tensorflow()


@keras_hub_export(
    "keras_hub.models.OpenAIPrivacyFilterPreprocessor",
)
class OpenAIPrivacyFilterPreprocessor(Preprocessor):
    """Preprocessor for OpenAI Privacy Filter token classification.

    This preprocessing layer tokenizes and pads/truncates input text into
    `token_ids` and `padding_mask` tensors suitable for the backbone. The
    model uses no start or end tokens, so the packed ids are the raw token
    ids followed only by padding.

    When per-token labels `y` are passed, they are padded or truncated to
    the same `sequence_length` (pad label `0`). When `sample_weight` is not
    provided, it defaults to `padding_mask` so that padded positions do not
    contribute to the loss. When an explicit `sample_weight` is provided,
    per-token weights are padded or truncated to `sequence_length`;
    a per-sample weight is broadcast over tokens and multiplied by the
    padding mask.

    The pad token is `<|endoftext|>`, as in the Hugging Face checkpoint. A
    literal `<|endoftext|>` inside the input text therefore receives the pad
    id and is masked out.

    Args:
        tokenizer: A `OpenAIPrivacyFilterTokenizer` instance.
        sequence_length: int. The length of the packed inputs.
            Defaults to `512`.
        truncate: string. The algorithm to truncate a list of batched
            segments. Defaults to `"round_robin"`.
    """

    backbone_cls = OpenAIPrivacyFilterBackbone
    tokenizer_cls = OpenAIPrivacyFilterTokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self.y_packer = None
        self._sequence_length = sequence_length
        self.truncate = truncate

    def build(self, input_shape):
        super().build(input_shape)
        # This encoder model uses no special start/end tokens (the HF
        # tokenizer adds none), so pack the raw token ids and only pad.
        # `MultiSegmentPacker` needs empty lists here; `None` raises.
        pad_id = self.tokenizer.pad_token_id
        self.packer = MultiSegmentPacker(
            start_value=[],
            end_value=[],
            pad_value=pad_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )
        self.y_packer = MultiSegmentPacker(
            start_value=[],
            end_value=[],
            pad_value=0,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )

    def _pad_sample_weight(self, sample_weight, padding_mask, sequence_length):
        """Align an explicit `sample_weight` with the packed `y`.

        A per-sample weight (one rank below `padding_mask`) is broadcast over
        the tokens and multiplied by the padding mask. A per-token weight is
        padded or truncated to `sequence_length`, like `y`, and is otherwise
        left as given. `MultiSegmentPacker` is float-preserving on the TF
        path but casts to `int32` on the Python path, so the Python path pads
        by hand.
        """
        sequence_length = sequence_length or self.sequence_length
        if self._use_tf_workflow():
            sw = convert_preprocessing_inputs(sample_weight)
            if sw.shape.rank == padding_mask.shape.rank - 1:
                return sw[..., None] * tf.cast(padding_mask, sw.dtype)
            sw, _ = self.y_packer((sw,), sequence_length=sequence_length)
            return sw
        # The packer returns backend tensors on the Python path; work in numpy
        # and convert back the same way the packer does.
        padding_mask = keras.ops.convert_to_numpy(padding_mask)
        # A ragged list of lists has `np.ndim == 1`; detect it by its first
        # element instead.
        is_nested = isinstance(sample_weight, (list, tuple)) and (
            len(sample_weight) > 0
            and isinstance(sample_weight[0], (list, tuple, np.ndarray))
        )
        rank = 2 if is_nested else np.ndim(sample_weight)
        if rank == padding_mask.ndim - 1:
            sw = np.asarray(sample_weight, dtype="float32")
            sw = sw[..., None] * padding_mask.astype(sw.dtype)
            return convert_preprocessing_outputs_python(sw)
        rows = [sample_weight] if padding_mask.ndim == 1 else sample_weight
        padded = []
        for row in rows:
            row = list(row)[:sequence_length]
            padded.append(row + [0.0] * (sequence_length - len(row)))
        sw = np.asarray(padded, dtype="float32")
        sw = sw[0] if padding_mask.ndim == 1 else sw
        return convert_preprocessing_outputs_python(sw)

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        x = x if isinstance(x, tuple) else (x,)
        x = tuple(self.tokenizer(segment) for segment in x)
        token_ids, _ = self.packer(x, sequence_length=sequence_length)
        padding_mask = token_ids != self.tokenizer.pad_token_id
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if y is not None:
            y, _ = self.y_packer((y,), sequence_length=sequence_length)
            if sample_weight is None:
                sample_weight = padding_mask
            else:
                sample_weight = self._pad_sample_weight(
                    sample_weight, padding_mask, sequence_length
                )
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        x = x if isinstance(x, tuple) else (x,)
        x = tuple(self.tokenizer(segment) for segment in x)
        token_ids, _ = self.packer(x, sequence_length=sequence_length)
        padding_mask = compute_padding_mask(
            token_ids, self.tokenizer.pad_token_id
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if y is not None:
            y, _ = self.y_packer((y,), sequence_length=sequence_length)
            if sample_weight is None:
                sample_weight = padding_mask
            else:
                sample_weight = self._pad_sample_weight(
                    sample_weight, padding_mask, sequence_length
                )
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if self._use_tf_workflow():
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

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        if self.packer is not None:
            self.packer.sequence_length = value
        if getattr(self, "y_packer", None) is not None:
            self.y_packer.sequence_length = value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "truncate": self.truncate,
            }
        )
        return config
