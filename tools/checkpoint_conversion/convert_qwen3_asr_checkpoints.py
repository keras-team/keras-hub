import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags

torch.manual_seed(123)
device = torch.device("cpu")
torch.set_default_device(device)


import keras_hub  # noqa: E402

PRESET_MAP = {
    "qwen3_asr_0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3_asr_1.7b": "Qwen/Qwen3-ASR-1.7B",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def test_model(keras_hub_backbone):
    """Verify backbone loads and produces the expected output shape."""
    audio_features = np.random.randn(1, 800, 128).astype("float32")
    token_ids = np.array([[1, 2, 3]], dtype="int32")
    # 800 -> 100 after 3x stride-2 downsampling. 100 + 3 = 103 total.
    padding_mask = np.array([[1] * 100 + [1, 1, 1]], dtype="int32")

    inputs = {
        "audio_features": audio_features,
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }
    output = keras_hub_backbone(inputs)
    print(f"Output shape: {output.shape}")
    expected_seq_len = 103
    assert output.shape[1] == expected_seq_len, (
        f"Expected seq_len={expected_seq_len}, got {output.shape[1]}"
    )
    print("-> Shape test passed!")


def main(_):
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    print(f"Loading Qwen3-ASR backbone from hf://{hf_preset}...")
    keras_hub_backbone = keras_hub.models.Qwen3ASRBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_tokenizer = keras_hub.models.Qwen3ASRTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("-> Model and tokenizer loaded")

    test_model(keras_hub_backbone)

    audio_converter = keras_hub.layers.Qwen3ASRAudioConverter()
    preprocessor = keras_hub.models.Qwen3ASRAudioToTextPreprocessor(
        audio_converter=audio_converter,
        tokenizer=keras_hub_tokenizer,
    )
    asr_model = keras_hub.models.Qwen3ASRAudioToText(
        backbone=keras_hub_backbone,
        preprocessor=preprocessor,
    )

    asr_model.save_to_preset(f"./{preset}")
    print(f"-> Saved to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
