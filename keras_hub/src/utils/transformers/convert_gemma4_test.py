import json
import os
import tempfile

from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
    Gemma4CausalLMPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_gemma4 import (
    load_preprocessor_config,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    load_video_converter_config,
)

_VIDEO_PROCESSOR = {
    "num_frames": 8,
    "max_soft_tokens": 70,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "do_resize": True,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


class ConvertGemma4Test(TestCase):
    def _write_preset(self, config, processor_config=None):
        """A local HF-layout preset: config.json, tokenizer.json and,
        when given, processor_config.json."""
        preset = tempfile.mkdtemp(dir=self.get_temp_dir())
        with open(os.path.join(preset, "config.json"), "w") as f:
            json.dump(config, f)
        # Minimal byte-fallback vocab so the tokenizer converter can build:
        # the three standard tokens plus `<|video|>`, which Gemma4Tokenizer
        # always registers.
        base = ["<pad>", "<eos>", "<bos>", "<unk>", "▁", "a", "<|video|>"]
        base += [f"<0x{i:02X}>" for i in range(256)]
        vocab = {tok: i for i, tok in enumerate(base)}
        added = [
            {"id": vocab[t], "content": t, "special": True}
            for t in base
            if t.startswith("<") and t.endswith(">") and len(t) > 1
        ]
        with open(os.path.join(preset, "tokenizer.json"), "w") as f:
            tokenizer_json = {
                "model": {"vocab": vocab, "merges": []},
                "added_tokens": added,
            }
            json.dump(tokenizer_json, f)
        if processor_config is not None:
            with open(os.path.join(preset, "processor_config.json"), "w") as f:
                json.dump(processor_config, f)
        return preset

    def test_text_only_preset_without_processor_config(self):
        # `gemma4_text` checkpoints ship no processor_config.json. The video
        # hooks must degrade to None / {} like the image and audio hooks do,
        # and the preprocessor must load without any converters.
        config = {"model_type": "gemma4_text", "vocab_size": 64}
        preset = self._write_preset(config)
        self.assertIsNone(load_video_converter_config(preset, config))
        self.assertEqual(load_preprocessor_config(preset, config), {})
        preprocessor = Gemma4CausalLMPreprocessor.from_preset(preset)
        self.assertIsNone(preprocessor.image_converter)
        self.assertIsNone(preprocessor.audio_converter)
        self.assertIsNone(preprocessor.video_converter)

    def test_video_preset_reads_video_processor(self):
        config = {"model_type": "gemma4", "vision_config": {}}
        preset = self._write_preset(
            config, processor_config={"video_processor": _VIDEO_PROCESSOR}
        )
        video_kwargs = load_video_converter_config(preset, config)
        self.assertEqual(video_kwargs["num_frames"], 8)
        self.assertEqual(video_kwargs["max_soft_tokens"], 70)
        self.assertEqual(video_kwargs["pooling_kernel_size"], 3)
        self.assertEqual(
            load_preprocessor_config(preset, config),
            {
                "num_frames_per_video": 8,
                "num_vision_tokens_per_frame": 70,
                "video_fps": 24.0,
            },
        )
        # No `video_processor` section: not a video model.
        preset = self._write_preset(config, processor_config={})
        self.assertIsNone(load_video_converter_config(preset, config))
        self.assertEqual(load_preprocessor_config(preset, config), {})
