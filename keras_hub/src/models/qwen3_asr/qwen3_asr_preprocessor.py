import keras
import numpy as np
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

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _get_audio_token_length(audio_lengths, n_window=50):
    chunk_len = n_window * 2
    remainder = audio_lengths % chunk_len
    # We use numpy/python math here as it is used in preprocessor
    feat_lengths = np.where(remainder > 0, (remainder - 1) // 2 + 1, 0)
    per_chunk_tokens = np.where(
        feat_lengths > 0, (feat_lengths - 1) // 2 + 1, 0
    )
    token_lengths = (
        (per_chunk_tokens - 1) // 2 + 1 + (audio_lengths // chunk_len) * 13
    )
    return token_lengths


def _get_audio_token_length_tf(audio_lengths, n_window=50):
    chunk_len = n_window * 2
    remainder = audio_lengths % chunk_len
    feat_lengths = tf.where(remainder > 0, (remainder - 1) // 2 + 1, 0)
    per_chunk_tokens = tf.where(
        feat_lengths > 0, (feat_lengths - 1) // 2 + 1, 0
    )
    token_lengths = (
        (per_chunk_tokens - 1) // 2 + 1 + (audio_lengths // chunk_len) * 13
    )
    return token_lengths


@keras_hub_export("keras_hub.models.Qwen3ASRPreprocessor")
class Qwen3ASRPreprocessor(CausalLMPreprocessor):
    backbone_cls = Qwen3ASRBackbone
    tokenizer_cls = Qwen3ASRTokenizer
    audio_converter_cls = Qwen3ASRAudioConverter

    def __init__(
        self,
        tokenizer,
        audio_converter=None,
        sequence_length=1024,
        add_start_token=False,
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

        # Special tokens
        self.audio_bos_token = "<|audio_start|>"
        self.audio_eos_token = "<|audio_end|>"
        self.audio_pad_token = "<|audio_pad|>"

    def build(self, input_shape):
        start_value = self.tokenizer.start_token_id
        if start_value is None:
            start_value = 0
        end_value = self.tokenizer.end_token_id
        if end_value is None:
            end_value = 0
        pad_value = self.tokenizer.pad_token_id
        if pad_value is None:
            pad_value = 0

        self.packer = MultiSegmentPacker(
            start_value=start_value,
            end_value=end_value,
            pad_value=pad_value,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    def _process_audio_and_text(self, x, sequence_length, add_end_token=None):
        if add_end_token is None:
            add_end_token = self.add_end_token if "responses" in x else False

        audios = x["audio"]
        prompts = x["prompts"]

        is_tf_tensor = tf is not None and isinstance(prompts, tf.Tensor)
        if is_tf_tensor:
            is_batched = prompts.shape.rank > 0
            if not is_batched:
                prompts = tf.expand_dims(prompts, 0)
                if isinstance(audios, (tf.Tensor, tf.RaggedTensor)):
                    audios = tf.expand_dims(audios, 0)
        else:
            is_batched = not isinstance(prompts, str)
            if not is_batched:
                prompts = [prompts]
                if isinstance(audios, np.ndarray):
                    if len(audios.shape) == 1:
                        audios = [audios]
                elif isinstance(audios, list):
                    if len(audios) > 0 and isinstance(audios[0], (int, float)):
                        audios = [audios]
                else:
                    audios = [audios]

        if self.audio_converter:
            audio_mel = self.audio_converter(audios)
            if is_tf_tensor:
                if isinstance(audios, tf.RaggedTensor):
                    raw_lengths = audios.row_lengths()
                else:
                    raw_lengths = tf.fill(
                        (tf.shape(audios)[0],), tf.shape(audios)[1]
                    )

                mel_lengths = (
                    tf.cast(raw_lengths, tf.int32)
                    // self.audio_converter.stride
                )
                max_mel_len = (
                    self.audio_converter.num_samples
                    // self.audio_converter.stride
                )
                mel_lengths = tf.minimum(mel_lengths, max_mel_len)

                max_len = tf.shape(audio_mel)[1]
                audio_mask = tf.sequence_mask(
                    mel_lengths, maxlen=max_len, dtype=tf.int32
                )

                num_audio_tokens = _get_audio_token_length_tf(
                    mel_lengths, self.audio_converter.n_window
                )
            else:
                raw_lengths = [len(a) for a in audios]
                mel_lengths = [
                    l // self.audio_converter.stride for l in raw_lengths
                ]
                mel_lengths = np.array(mel_lengths, dtype=np.int32)

                max_mel_len = (
                    self.audio_converter.num_samples
                    // self.audio_converter.stride
                )
                mel_lengths = np.minimum(mel_lengths, max_mel_len)

                max_len = audio_mel.shape[1]
                audio_mask = np.zeros(
                    (len(raw_lengths), max_len), dtype=np.int32
                )
                for i, l in enumerate(mel_lengths):
                    audio_mask[i, :l] = 1

                num_audio_tokens = _get_audio_token_length(
                    mel_lengths, self.audio_converter.n_window
                )
        else:
            audio_mel = None
            audio_mask = None
            if is_tf_tensor:
                num_audio_tokens = tf.zeros(
                    (tf.shape(prompts)[0],), dtype=tf.int32
                )
            else:
                num_audio_tokens = [0] * len(prompts)

        if is_tf_tensor:
            batch_size = tf.shape(num_audio_tokens)[0]
            pads = tf.fill((batch_size,), self.audio_pad_token)
            repeated_pads = tf.repeat(pads, num_audio_tokens)
            ragged_pads = tf.RaggedTensor.from_row_lengths(
                repeated_pads, num_audio_tokens
            )
            padded_str = tf.strings.reduce_join(ragged_pads, axis=-1)

            audio_str = tf.where(
                num_audio_tokens > 0,
                self.audio_bos_token + padded_str + self.audio_eos_token,
                "",
            )

            def process_sample(args):
                prompt, a_str, num_tokens = args
                segments = tf.strings.split(prompt, sep="<audio>")
                num_segments = tf.shape(segments)[0]

                def replace_fn():
                    num_seg = tf.shape(segments)[0]
                    separators = tf.fill((num_seg - 1,), a_str)
                    indices_seg = tf.range(0, 2 * num_seg, 2)
                    indices_sep = tf.range(1, 2 * num_seg - 1, 2)
                    interleaved = tf.dynamic_stitch(
                        [indices_seg, indices_sep], [segments, separators]
                    )
                    return tf.strings.reduce_join(interleaved, separator="")

                def prepend_fn():
                    return tf.cond(
                        num_tokens > 0,
                        lambda: tf.strings.join(
                            [a_str, prompt], separator="\n"
                        ),
                        lambda: prompt,
                    )

                return tf.cond(num_segments > 1, replace_fn, prepend_fn)

            formatted_prompts = tf.map_fn(
                process_sample,
                elems=(prompts, audio_str, num_audio_tokens),
                fn_output_signature=tf.string,
            )
        else:
            formatted_prompts = []
            for prompt, num_tokens in zip(prompts, num_audio_tokens):
                if num_tokens > 0:
                    audio_str = (
                        self.audio_bos_token
                        + self.audio_pad_token * num_tokens
                        + self.audio_eos_token
                    )
                    if "<audio>" in prompt:
                        formatted_prompt = prompt.replace("<audio>", audio_str)
                    else:
                        formatted_prompt = audio_str + "\n" + prompt
                else:
                    formatted_prompt = prompt
                formatted_prompts.append(formatted_prompt)

        tokenized_prompts = self.tokenizer(formatted_prompts)

        if "responses" in x:
            responses = x["responses"]
            if is_tf_tensor and not is_batched:
                responses = tf.expand_dims(responses, 0)
            elif not is_batched and isinstance(responses, str):
                responses = [responses]
            tokenized_responses = self.tokenizer(responses)
            segments = (tokenized_prompts, tokenized_responses)
        else:
            segments = (tokenized_prompts,)

        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=add_end_token,
        )

        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1

        if not is_batched:
            token_ids = ops.squeeze(token_ids, axis=0)
            padding_mask = ops.squeeze(padding_mask, axis=0)
            response_mask = ops.squeeze(response_mask, axis=0)
            if audio_mel is not None:
                audio_mel = ops.squeeze(audio_mel, axis=0)
                audio_mask = audio_mask[0]

        output = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if audio_mel is not None:
            output["audio_mel"] = audio_mel
            output["audio_mask"] = ops.convert_to_tensor(audio_mask)

        return output, response_mask

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length
        is_training = "responses" in x
        pack_seq_len = sequence_length + 1 if is_training else sequence_length

        features, response_mask = self._process_audio_and_text(x, pack_seq_len)

        if is_training:
            # Extract y and sample_weight from UNTRUNCATED features
            y_out = features["token_ids"][..., 1:]
            sample_weight_out = response_mask[..., 1:]

            # Truncate features for x
            features["token_ids"] = features["token_ids"][..., :-1]
            features["padding_mask"] = features["padding_mask"][..., :-1]

            x_out = features
            return keras.utils.pack_x_y_sample_weight(
                x_out, y_out, sample_weight_out
            )
        else:
            return features

    @preprocessing_function
    def _call_tf(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
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
        sequence_length = sequence_length or self.sequence_length
        features, _ = self._process_audio_and_text(
            x, sequence_length, add_end_token=False
        )
        return features

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        return self._generate_preprocess_python(
            x, sequence_length=sequence_length
        )

    def generate_preprocess(self, x, sequence_length=None):
        if not self.built:
            self.build(None)
        if not self._allow_python_workflow or in_tf_function():
            return self._generate_preprocess_tf(x, sequence_length)
        else:
            return self._generate_preprocess_python(x, sequence_length)
