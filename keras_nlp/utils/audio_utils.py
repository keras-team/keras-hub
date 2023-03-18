# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import tensorflow as tf

try:
    import ffmpeg
except ImportError:
    ffmpeg = None


def load_audio(file, sample_rate=16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary.
    Adapted from https://github.com/openai/whisper/blob/v20230314/whisper/audio.py#L26.

    Note: Tried using `tensorflow-io`, but it apparently uses a different
    function for resampling. Hence, sticking to `ffmpeg-python` for now.
    """
    if ffmpeg is None:
        raise ImportError(
            "keras_nlp.utils.audio_utils.load_audio` requires the ffmpeg-python` "
            "package. Please install it with `pip install ffmpeg-python`."
        )

    try:
        # This launches a subprocess to decode audio while down-mixing and
        # resampling as necessary.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output(
                "-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # Convert to TensorFlow tensor.
    out = tf.constant(out)
    return out
