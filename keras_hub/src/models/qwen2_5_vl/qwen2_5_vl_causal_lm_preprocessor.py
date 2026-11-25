import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_tokenizer import (
    Qwen2_5VLTokenizer,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_video_converter import (
    Qwen2_5VLVideoConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)


@keras_hub_export("keras_hub.models.Qwen2_5VLCausalLMPreprocessor")
class Qwen2_5VLCausalLMPreprocessor(Qwen2VLCausalLMPreprocessor):
    """Qwen2.5-VL Causal LM preprocessor with multimodal support.

    Behaves identically to ``Qwen2VLCausalLMPreprocessor``, with one
    addition: videos use time-aware temporal M-RoPE positions. Each video
    gets ``time_interval = tokens_per_second * second_per_grid_ts`` and a
    token in temporal patch ``f`` receives temporal position
    ``int(f * time_interval)`` (matching HF ``get_rope_index`` — note the
    truncation happens per-frame, so fractional intervals produce
    non-uniform steps).

    ``second_per_grid_ts`` is supplied per video (in prompt order, matching
    the video grids) via the optional ``"second_per_grid_ts"`` key in the
    input dict — e.g. ``temporal_patch_size / fps`` as computed by HF's
    processor from video metadata. When absent, every video uses
    ``second_per_grid_ts = 1``, i.e. ``time_interval = tokens_per_second``
    (HF's fallback when fps metadata is unavailable).

    Args:
        tokenizer: A ``Qwen2_5VLTokenizer`` instance.
        image_converter: A ``Qwen2_5VLImageConverter`` instance, or ``None``
            for text-only mode. Defaults to ``None``.
        video_converter: A ``Qwen2_5VLVideoConverter`` instance, or
            ``None``. When unset, video preprocessing falls back to
            ``image_converter`` (shared, so bounds mutations propagate).
            Defaults to ``None``.
        tokens_per_second: int. Temporal position scale per second of
            video (HF ``vision_config.tokens_per_second``). Defaults to
            ``4``.
        sequence_length: int. Total padded sequence length. Defaults to
            ``1024``.
        add_start_token: bool. Prepend BOS token. Defaults to ``False``.
        add_end_token: bool. Append EOS token. Defaults to ``True``.

    Examples:

    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.Qwen2_5VLCausalLMPreprocessor.from_preset(
        "qwen2_5_vl_3b_instruct",
    )

    # Tokenize and pad a single sequence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and pad a batch of sequences.
    preprocessor(["The quick brown fox jumped.", "The fox slept."])

    # Multimodal input: image plus chat prompt with a vision placeholder.
    preprocessor(
        {
            "prompts": ["<|vision_start|><|image_pad|><|vision_end|>"
                        "Describe this image.<|im_end|>"],
            "images": [np.random.randint(0, 256, size=(224, 224, 3))],
        }
    )
    ```
    """

    backbone_cls = Qwen2_5VLBackbone
    tokenizer_cls = Qwen2_5VLTokenizer
    image_converter_cls = Qwen2_5VLImageConverter
    video_converter_cls = Qwen2_5VLVideoConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        video_converter=None,
        tokens_per_second=4,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            image_converter=image_converter,
            video_converter=video_converter,
            **kwargs,
        )
        self.tokens_per_second = tokens_per_second

    def _temporal_position(self, frame, is_video, media_index, sgt):
        if not is_video:
            return frame
        second_per_grid_ts = 1.0
        if sgt is not None and media_index < len(sgt):
            second_per_grid_ts = float(sgt[media_index])
        time_interval = self.tokens_per_second * second_per_grid_ts
        # Temporal index is the float32 product of frame and interval,
        # truncated.
        return int(np.float32(frame) * np.float32(time_interval))

    def get_config(self):
        config = super().get_config()
        config.update({"tokens_per_second": self.tokens_per_second})
        return config
