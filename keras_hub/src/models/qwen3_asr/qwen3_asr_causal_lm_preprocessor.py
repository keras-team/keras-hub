import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.Qwen3ASRCausalLMPreprocessor")
class Qwen3ASRCausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen3-ASR task preprocessor.

    Prepares inputs for causal language model generation tasks with interleaved
    audio.
    """

    backbone_cls = Qwen3ASRBackbone
    tokenizer_cls = Qwen3ASRTokenizer
    audio_converter_cls = Qwen3ASRAudioConverter

    def __init__(
        self,
        tokenizer,
        audio_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        audio_placeholder="<|AUDIO|>",
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
        self.audio_placeholder = audio_placeholder

    def build(self, input_shape):
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id
            if self.tokenizer.start_token_id is not None
            else [],
            end_value=self.tokenizer.end_token_id
            if self.tokenizer.end_token_id is not None
            else [],
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length
        audio = x.get("audio", None)
        prompts = x["prompts"]
        responses = x.get("responses", None)

        # 1. Process Audio
        audio_mel = None
        audio_mel_mask = None
        num_audio_tokens = 0
        if audio is not None and self.audio_converter is not None:
            audio_mel = self.audio_converter(audio)
            T = ops.shape(audio_mel)[1]
            audio_mel_mask = ops.ones(
                (ops.shape(audio_mel)[0], T), dtype="int32"
            )

            chunk_len = 100
            num_chunks = T // chunk_len
            num_audio_tokens = num_chunks * 13

        # 2. Expand Prompts with placeholders
        if num_audio_tokens > 0:
            if isinstance(prompts, str):
                prompts = prompts.replace(
                    self.audio_placeholder,
                    self.audio_placeholder * num_audio_tokens,
                )
            elif isinstance(prompts, (list, tuple)):
                prompts = [
                    p.replace(
                        self.audio_placeholder,
                        self.audio_placeholder * num_audio_tokens,
                    )
                    for p in prompts
                ]

        # 3. Tokenize
        prompts = self.tokenizer(prompts)
        if responses is not None:
            responses = self.tokenizer(responses)
        else:
            responses = []

        # 4. Pack
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1

        # 5. Find Audio Indices
        audio_token_id = self.tokenizer.audio_token_id
        indices_mask = token_ids == audio_token_id

        where_result = ops.where(indices_mask)
        if isinstance(where_result, (tuple, list)):
            audio_indices = where_result[1]
        else:
            audio_indices = where_result[:, 1]  # seq_idx

        B = ops.shape(token_ids)[0]
        audio_indices = ops.reshape(audio_indices, (B, -1))

        x_out = {
            "token_ids": token_ids[..., :-1],
            "response_mask": response_mask[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }
        if audio_mel is not None:
            x_out["audio_mel"] = audio_mel
            x_out["audio_mel_mask"] = audio_mel_mask
            x_out["audio_indices"] = audio_indices

        y_out = token_ids[..., 1:]
        sample_weight_out = response_mask[..., 1:]

        return keras.utils.pack_x_y_sample_weight(
            x_out, y_out, sample_weight_out
        )

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        return self._call_python(x, y, sample_weight, sequence_length)

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self._allow_python_workflow or in_tf_function():
            return self._call_tf(x, y, sample_weight, sequence_length)
        else:
            return self._call_python(x, y, sample_weight, sequence_length)
