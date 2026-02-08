import keras

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
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export(
    "keras_hub.models.Qwen3OmniCausalLMPreprocessor",
)
class Qwen3OmniCausalLMPreprocessor(CausalLMPreprocessor):
    """Multimodal preprocessor for Qwen3-Omni CausalLM.

    Handles preprocessing for text, audio, and vision inputs.

    Args:
        tokenizer: Qwen3OmniTokenizer instance.
        audio_converter: Qwen3OmniAudioConverter instance (optional).
        image_converter: Qwen3OmniImageConverter instance (optional).
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
    }
    output = preprocessor(x)
    ```
    """

    backbone_cls = Qwen3OmniBackbone
    tokenizer_cls = Qwen3OmniTokenizer
    audio_converter_cls = Qwen3OmniAudioConverter
    image_converter_cls = Qwen3OmniImageConverter

    def __init__(
        self,
        tokenizer,
        audio_converter=None,
        image_converter=None,
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

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        # Use MultiSegmentPacker for dict input with prompts/responses
        # Use StartEndPacker from parent for simple text input
        super().build(input_shape)
        # Handle None start/end tokens by using empty lists
        start_val = (
            self.tokenizer.start_token_id
            if self.tokenizer.start_token_id is not None
            else []
        )
        end_val = (
            self.tokenizer.end_token_id
            if self.tokenizer.end_token_id is not None
            else []
        )
        self.multi_packer = MultiSegmentPacker(
            start_value=start_val,
            end_value=end_val,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # Handle both dict input (multimodal) and string input (text-only)
        if isinstance(x, dict):
            prompts = x["prompts"]
            responses = x.get("responses", prompts)
        else:
            # Text-only mode: tokenize directly
            x = self.tokenizer(x)
            token_ids, padding_mask = self.packer(
                x,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            # The last token does not have a next token, so we truncate it out.
            result_x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            # Target `y` will be the next token.
            result_y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
            return keras.utils.pack_x_y_sample_weight(
                result_x, result_y, sample_weight
            )

        # Tokenize text
        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        # Process multimodal inputs
        audio_features = None
        if "audio" in x and self.audio_converter:
            audio_features = self.audio_converter(x["audio"])

        pixel_values = None
        if "images" in x and self.image_converter:
            pixel_values = self.image_converter(x["images"])

        # Pad with one extra token to account for the truncation below.
        token_ids, segment_ids = self.multi_packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1

        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }

        # Add multimodal inputs if present
        if audio_features is not None:
            x["audio_features"] = audio_features
        if pixel_values is not None:
            x["pixel_values"] = pixel_values

        # Target `y` will be the next token.
        y = token_ids[..., 1:]
        # Only compute the loss for labels in the response.
        sample_weight = response_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert inputs to integer token input for generation.

        Handles text, audio, and vision inputs for multimodal generation.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        if not self.built:
            self.build(None)

        # Handle both dict input (multimodal) and string input (text-only)
        if not isinstance(x, dict):
            # Text-only mode: tokenize directly
            x = self.tokenizer(x)
            token_ids, padding_mask = self.packer(
                x, sequence_length=sequence_length, add_end_value=False
            )
            return {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            }

        prompts = x["prompts"]
        prompts = self.tokenizer(prompts)

        # Process multimodal inputs
        audio_features = None
        if "audio" in x and self.audio_converter:
            audio_features = self.audio_converter(x["audio"])

        pixel_values = None
        if "images" in x and self.image_converter:
            pixel_values = self.image_converter(x["images"])

        if "responses" in x:
            responses = self.tokenizer(x["responses"])
            segments = (prompts, responses)
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

        # Add multimodal inputs if present
        if audio_features is not None:
            result["audio_features"] = audio_features
        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result
