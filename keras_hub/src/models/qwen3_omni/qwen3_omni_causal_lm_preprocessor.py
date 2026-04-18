import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_converter import (
    Qwen3OmniAudioConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_image_converter import (
    Qwen3OmniImageConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_tokenizer import (
    Qwen3OmniTokenizer,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_video_converter import (
    Qwen3OmniVideoConverter,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export(
    "keras_hub.models.Qwen3OmniCausalLMPreprocessor",
)
class Qwen3OmniCausalLMPreprocessor(CausalLMPreprocessor):
    """Multimodal preprocessor for Qwen3-Omni CausalLM.

    Handles preprocessing for text, audio, image, and video inputs.

    Args:
        tokenizer: Qwen3OmniTokenizer instance.
        audio_converter: Qwen3OmniAudioConverter instance (optional).
        image_converter: Qwen3OmniImageConverter instance (optional).
        video_converter: Qwen3OmniVideoConverter instance (optional).
        sequence_length: int. Maximum sequence length. Defaults to 1024.
        add_start_token: bool. Whether to add start token. Defaults to True.
        add_end_token: bool. Whether to add end token. Defaults to True.
        **kwargs: Additional layer arguments.

    Examples:
    ```python
    # Text-only preprocessing
    preprocessor = keras_hub.models.Qwen3OmniCausalLMPreprocessor.from_preset(
        "qwen3_omni_instruct"
    )
    x = {"prompts": "Hello", "responses": "Hi there!"}
    output = preprocessor(x)

    # Multimodal preprocessing
    x = {
        "prompts": "What is in this image?",
        "responses": "A cat",
        "images": image_array,
        "audio": audio_array,
        "video": video_frames_array,
    }
    output = preprocessor(x)
    ```
    """

    backbone_cls = Qwen3OmniBackbone
    tokenizer_cls = Qwen3OmniTokenizer
    audio_converter_cls = Qwen3OmniAudioConverter
    image_converter_cls = Qwen3OmniImageConverter
    video_converter_cls = Qwen3OmniVideoConverter

    def __init__(
        self,
        tokenizer,
        audio_converter=None,
        image_converter=None,
        video_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.audio_converter = audio_converter
        self.image_converter = image_converter
        self.video_converter = video_converter

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        super().build(input_shape)
        self.multi_packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id or [],
            end_value=self.tokenizer.end_token_id or [],
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )

    def _process_multimodal_inputs(self, x):
        """Extract and convert audio, image, and video inputs from a dict.

        Image and video patches/grids are concatenated along the visual
        axis — the backbone's ``_masked_scatter`` uses a combined
        image+video token mask and the vision encoder handles both in
        one pass.
        """
        audio_features = None
        if "audio" in x and self.audio_converter:
            audio_features = self.audio_converter(x["audio"])

        image_out = None
        if "images" in x and self.image_converter:
            image_out = self.image_converter(x["images"])
        video_out = None
        if "video" in x and self.video_converter:
            video_out = self.video_converter(x["video"])

        pixel_values = None
        grid_thw = None
        if image_out is not None and video_out is not None:
            pixel_values = ops.concatenate(
                [image_out["patches"], video_out["patches"]], axis=0
            )
            grid_thw = ops.stack(
                [image_out["grid_thw"], video_out["grid_thw"]], axis=0
            )
        elif image_out is not None:
            pixel_values = image_out["patches"]
            grid_thw = ops.expand_dims(image_out["grid_thw"], axis=0)
        elif video_out is not None:
            pixel_values = video_out["patches"]
            grid_thw = ops.expand_dims(video_out["grid_thw"], axis=0)

        return audio_features, pixel_values, grid_thw

    def _add_multimodal_to_output(
        self, output, audio_features, pixel_values, grid_thw
    ):
        """Attach multimodal features to the output dict if present."""
        if audio_features is not None:
            output["audio_features"] = audio_features
        if pixel_values is not None:
            output["pixel_values"] = pixel_values
        if grid_thw is not None:
            output["grid_thw"] = grid_thw

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # Text-only input (string or tensor)
        if not isinstance(x, dict):
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

        # Multimodal dict input
        prompts = self.tokenizer(x["prompts"])
        audio_features, pixel_values, grid_thw = (
            self._process_multimodal_inputs(x)
        )
        responses_text = x.get("responses", None)

        if responses_text is not None:
            responses = self.tokenizer(responses_text)
            # Pack prompt + response with one extra token for label shift.
            token_ids, segment_ids = self.multi_packer(
                (prompts, responses),
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            padding_mask = token_ids != self.tokenizer.pad_token_id
            response_mask = segment_ids == 1

            # Truncate last token (no next-token target for it).
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            self._add_multimodal_to_output(
                x, audio_features, pixel_values, grid_thw
            )

            y = token_ids[..., 1:]
            sample_weight = response_mask[..., 1:]
        else:
            # No responses — single-segment next-token prediction.
            token_ids, padding_mask = self.packer(
                prompts,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            self._add_multimodal_to_output(
                x, audio_features, pixel_values, grid_thw
            )

            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]

        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert inputs to integer token input for generation.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        if not self.built:
            self.build(None)

        # Text-only input (string or tensor)
        if not isinstance(x, dict):
            x = self.tokenizer(x)
            token_ids, padding_mask = self.packer(
                x, sequence_length=sequence_length, add_end_value=False
            )
            return {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            }

        # Multimodal dict input
        prompts = self.tokenizer(x["prompts"])
        audio_features, pixel_values, grid_thw = (
            self._process_multimodal_inputs(x)
        )

        if "responses" in x:
            segments = (prompts, self.tokenizer(x["responses"]))
        else:
            segments = (prompts,)

        token_ids, segment_ids = self.multi_packer(
            segments,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        result = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        self._add_multimodal_to_output(
            result, audio_features, pixel_values, grid_thw
        )
        return result
