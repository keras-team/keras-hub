import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import _to_host
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import tf


def _is_1d_audio_list(x):
    return (
        isinstance(x, list)
        and len(x) > 0
        and isinstance(x[0], (int, float, np.integer, np.floating))
    )


def _decode_prompt(prompt):
    if isinstance(prompt, np.ndarray) and prompt.ndim == 0:
        prompt = prompt.item()
    if isinstance(prompt, (bytes, np.bytes_)):
        return prompt.decode("utf-8")
    return prompt


def _get_audio_chunk_token_length(n_window):
    chunk_token_length = n_window * 2
    for _ in range(3):
        chunk_token_length = (chunk_token_length - 1) // 2 + 1
    return chunk_token_length


def _get_audio_token_length(audio_lengths, n_window=50):
    chunk_len = n_window * 2
    chunk_token_length = _get_audio_chunk_token_length(n_window)
    remainder = audio_lengths % chunk_len
    feat_lengths = (remainder - 1) // 2 + 1
    per_chunk_tokens = (feat_lengths - 1) // 2 + 1
    token_lengths = (
        (per_chunk_tokens - 1) // 2
        + 1
        + (audio_lengths // chunk_len) * chunk_token_length
    )
    return token_lengths


@keras_hub_export("keras_hub.models.Qwen3ASRCausalLMPreprocessor")
class Qwen3ASRCausalLMPreprocessor(CausalLMPreprocessor):
    """Preprocess Qwen3-ASR audio-text inputs for causal LM generation.

    The preprocessor turns raw audio and prompt text into the token IDs,
    attention masks, and audio features expected by the ASR backbone. Audio
    is converted to mel features, the prompt is augmented with the audio
    placeholder tokens, and optional responses are packed into a training
    sequence.

    Args:
        tokenizer: A `Qwen3ASRTokenizer` instance.
        audio_converter: A `Qwen3ASRAudioConverter` instance, optional. Audio
            converter used to compute mel features. Defaults to `None`.
        sequence_length: int, optional. Maximum length of packed token
            sequences. Defaults to `1024`.
        add_start_token: bool, optional. Whether to prepend the tokenizer start
            token. Defaults to `False`.
        add_end_token: bool, optional. Whether to append the tokenizer end
            token. Defaults to `True`.

    Example:
        ```python
        preprocessor = keras_hub.models.Qwen3ASRCausalLMPreprocessor(
            tokenizer=tokenizer,
            audio_converter=Qwen3ASRAudioConverter(),
        )
        x = {
            "audio": np.ones((16000,), dtype="float32"),
            "prompts": " airplane",
        }
        preprocessor(x)
        ```
    """

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
            start_value = []
        end_value = self.tokenizer.end_token_id
        if end_value is None:
            end_value = []
        pad_value = self.tokenizer.pad_token_id
        if pad_value is None:
            pad_value = 0

        self.packer = MultiSegmentPacker(
            start_value=start_value,
            end_value=end_value,
            pad_value=pad_value,
            sep_value=[],
            sequence_length=self.sequence_length,
            truncate="waterfall",
        )
        self.built = True

    def _process_audio_and_text(self, x, sequence_length, add_end_token=None):
        if add_end_token is None:
            add_end_token = self.add_end_token if "responses" in x else False

        audios = x.get("audio")
        prompts = x["prompts"]
        if self.audio_converter is not None and audios is None:
            raise ValueError(
                "`audio` must be provided when `audio_converter` is set."
            )

        is_tf_tensor = tf is not None and isinstance(prompts, tf.Tensor)
        if is_tf_tensor:
            audios = _to_host(audios)
            is_batched = prompts.shape.rank > 0
            if not is_batched:
                prompts = tf.expand_dims(prompts, 0)
                if isinstance(audios, (tf.Tensor, tf.RaggedTensor)):
                    audios = tf.expand_dims(audios, 0)
                elif _is_1d_audio_list(audios):
                    audios = [audios]
                elif isinstance(audios, np.ndarray):
                    if audios.ndim == 1:
                        audios = [audios]
                elif audios is not None and keras.ops.is_tensor(audios):
                    audios_np = convert_to_numpy(audios)
                    if audios_np.ndim == 2 and audios_np.shape[0] == 1:
                        audios = [audios_np[0]]
                    else:
                        audios = [audios_np]
                elif audios is not None and not isinstance(
                    audios, (list, tuple)
                ):
                    audios = [audios]
            elif audios is not None:
                if _is_1d_audio_list(audios) or (
                    isinstance(audios, np.ndarray) and audios.ndim == 1
                ):
                    audios = [audios]
        else:
            if isinstance(prompts, (bytes, np.bytes_)) or (
                isinstance(prompts, np.ndarray) and prompts.ndim == 0
            ):
                prompts = _decode_prompt(prompts)
            is_batched = not isinstance(prompts, str)
            if not is_batched:
                prompts = [prompts]
                if audios is not None:
                    audios = _to_host(audios)
                    if keras.ops.is_tensor(audios) or isinstance(
                        audios, np.ndarray
                    ):
                        audios_np = convert_to_numpy(audios)
                        if audios_np.ndim == 1:
                            audios = [audios_np]
                        elif audios_np.ndim == 2 and audios_np.shape[0] == 1:
                            audios = [audios_np[0]]
                        else:
                            audios = list(audios_np)
                    elif _is_1d_audio_list(audios):
                        audios = [audios]
                    elif isinstance(audios, (list, tuple)):
                        audios = [
                            convert_to_numpy(_to_host(a))
                            if keras.ops.is_tensor(a)
                            else a
                            for a in audios
                        ]
                    else:
                        audios = [audios]
            else:
                prompts = [_decode_prompt(p) for p in prompts]
                if audios is not None:
                    audios = _to_host(audios)
                    if keras.ops.is_tensor(audios) or isinstance(
                        audios, np.ndarray
                    ):
                        audios_np = convert_to_numpy(audios)
                        audios = (
                            [audios_np]
                            if audios_np.ndim == 1
                            else list(audios_np)
                        )
                    elif _is_1d_audio_list(audios):
                        audios = [audios]
                    else:
                        audios = [
                            convert_to_numpy(_to_host(a))
                            if keras.ops.is_tensor(a)
                            else a
                            for a in audios
                        ]

        if self.audio_converter:
            if is_tf_tensor and not isinstance(
                audios, (tf.Tensor, tf.RaggedTensor)
            ):
                if len(prompts) != len(audios):
                    raise ValueError(
                        f"Number of prompts ({len(prompts)}) must match "
                        f"number of audio clips ({len(audios)})."
                    )
                raw_lengths = tf.constant(
                    [len(a) for a in audios], dtype=tf.int32
                )
            elif not is_tf_tensor and len(prompts) != len(audios):
                raise ValueError(
                    f"Number of prompts ({len(prompts)}) must match "
                    f"number of audio clips ({len(audios)})."
                )

            audio_mel = self.audio_converter(audios)
            if is_tf_tensor:
                audio_mel = _to_host(audio_mel)
                if isinstance(audios, tf.RaggedTensor):
                    raw_lengths = audios.row_lengths()
                elif isinstance(audios, tf.Tensor):
                    raw_lengths = tf.fill(
                        (tf.shape(audios)[0],), tf.shape(audios)[1]
                    )
                raw_lengths = tf.maximum(
                    raw_lengths, self.audio_converter.min_length
                )

                mel_lengths = (
                    tf.cast(raw_lengths, tf.int32)
                    // self.audio_converter.stride
                )
                max_len = tf.shape(audio_mel)[1]
                audio_mask = tf.sequence_mask(
                    mel_lengths, maxlen=max_len, dtype=tf.int32
                )

                num_audio_tokens = _get_audio_token_length(
                    mel_lengths, self.audio_converter.n_window
                )
            else:
                audio_mel = convert_preprocessing_outputs_python(audio_mel)
                raw_lengths = [
                    max(len(a), self.audio_converter.min_length) for a in audios
                ]
                mel_lengths = np.array(
                    [l // self.audio_converter.stride for l in raw_lengths],
                    dtype=np.int32,
                )
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
                check_placeholders = tf.debugging.assert_less_equal(
                    num_segments,
                    2,
                    message=(
                        "Each prompt may contain at most one `<audio>` "
                        "placeholder."
                    ),
                )

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
                        lambda: tf.strings.join([a_str, prompt], separator=""),
                        lambda: prompt,
                    )

                with tf.control_dependencies([check_placeholders]):
                    return tf.cond(num_segments > 1, replace_fn, prepend_fn)

            formatted_prompts = tf.map_fn(
                process_sample,
                elems=(prompts, audio_str, num_audio_tokens),
                fn_output_signature=tf.string,
            )
        else:
            formatted_prompts = []
            for prompt, num_tokens in zip(prompts, num_audio_tokens):
                placeholder_count = prompt.count("<audio>")
                if placeholder_count > 1:
                    raise ValueError(
                        "Each prompt may contain at most one `<audio>` "
                        f"placeholder, got {placeholder_count}."
                    )
                if num_tokens > 0:
                    audio_str = (
                        self.audio_bos_token
                        + self.audio_pad_token * num_tokens
                        + self.audio_eos_token
                    )
                    if placeholder_count == 1:
                        formatted_prompt = prompt.replace("<audio>", audio_str)
                    else:
                        formatted_prompt = audio_str + prompt
                else:
                    formatted_prompt = prompt
                formatted_prompts.append(formatted_prompt)

        tokenized_prompts = self.tokenizer(formatted_prompts)

        if "responses" in x:
            responses = x["responses"]
            if is_tf_tensor and not is_batched:
                responses = tf.expand_dims(responses, 0)
            elif not is_tf_tensor:
                if not is_batched:
                    responses = [_decode_prompt(responses)]
                else:
                    responses = [_decode_prompt(r) for r in responses]
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

        if self.audio_converter is not None:
            audio_pad_id = self.tokenizer.token_to_id(self.audio_pad_token)
            model_token_ids = (
                token_ids[:, :-1] if "responses" in x else token_ids
            )
            if in_tf_function():
                actual_audio_tokens = tf.reduce_sum(
                    tf.cast(
                        tf.equal(_to_host(model_token_ids), audio_pad_id),
                        tf.int32,
                    ),
                    axis=-1,
                )
                check = tf.debugging.assert_equal(
                    actual_audio_tokens,
                    tf.cast(num_audio_tokens, tf.int32),
                    message=(
                        "Audio placeholder tokens `<|audio_pad|>` were "
                        "truncated by `sequence_length`. Increase "
                        "`sequence_length` so the full prompt and all audio "
                        "tokens fit."
                    ),
                )
                with tf.control_dependencies([check]):
                    token_ids = tf.identity(_to_host(token_ids))
            else:
                actual_audio_tokens = np.sum(
                    convert_to_numpy(model_token_ids) == audio_pad_id,
                    axis=-1,
                )
                expected_audio_tokens = convert_to_numpy(num_audio_tokens)
                if not np.array_equal(
                    actual_audio_tokens, expected_audio_tokens
                ):
                    raise ValueError(
                        "Audio placeholder tokens `<|audio_pad|>` were "
                        "truncated by `sequence_length` (expected "
                        f"{expected_audio_tokens.tolist()}, got "
                        f"{actual_audio_tokens.tolist()}). Increase "
                        "`sequence_length` so the full prompt and all audio "
                        "tokens fit."
                    )

        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1

        if not is_batched:
            if is_tf_tensor:
                token_ids = tf.reshape(_to_host(token_ids), (sequence_length,))
                padding_mask = tf.reshape(
                    _to_host(padding_mask), (sequence_length,)
                )
                response_mask = tf.reshape(
                    _to_host(response_mask), (sequence_length,)
                )
                if audio_mel is not None:
                    audio_mel = tf.squeeze(audio_mel, axis=0)
                    audio_mask = tf.squeeze(audio_mask, axis=0)
            else:
                token_ids = keras.ops.reshape(token_ids, (sequence_length,))
                padding_mask = keras.ops.reshape(
                    padding_mask, (sequence_length,)
                )
                response_mask = keras.ops.reshape(
                    response_mask, (sequence_length,)
                )
                if audio_mel is not None:
                    audio_mel = (
                        np.squeeze(audio_mel, axis=0)
                        if isinstance(audio_mel, np.ndarray)
                        else keras.ops.squeeze(audio_mel, axis=0)
                    )
                    audio_mask = np.squeeze(audio_mask, axis=0)

        output = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if audio_mel is not None:
            output["audio_mel"] = audio_mel
            output["audio_mask"] = audio_mask

        return convert_preprocessing_outputs_python((output, response_mask))

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

    def _generate_preprocess_python(self, x, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        features, _ = self._process_audio_and_text(
            x, sequence_length, add_end_token=False
        )
        return features
