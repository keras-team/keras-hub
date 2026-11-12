import json
import os
import tempfile

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
    def _write_preset(self, processor_config=None):
        """A local HF-layout preset: processor_config.json when given."""
        preset = tempfile.mkdtemp(dir=self.get_temp_dir())
        if processor_config is not None:
            with open(os.path.join(preset, "processor_config.json"), "w") as f:
                json.dump(processor_config, f)
        return preset

    def test_video_preset_reads_video_processor(self):
        # 1. `vision_config` absent (e.g. `gemma4_text`). The video hooks must
        # degrade to None / {} like the image and audio hooks do.
        config = {"model_type": "gemma4_text", "vocab_size": 64}
        preset = self._write_preset()
        self.assertIsNone(load_video_converter_config(preset, config))
        self.assertEqual(load_preprocessor_config(preset, config), {})

        # 2. `vision_config` present, `video_processor` present.
        config = {"model_type": "gemma4", "vision_config": {}}
        preset = self._write_preset(
            processor_config={"video_processor": _VIDEO_PROCESSOR}
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

        # 3. `vision_config` present, `video_processor` absent.
        preset = self._write_preset(processor_config={})
        self.assertIsNone(load_video_converter_config(preset, config))
        self.assertEqual(load_preprocessor_config(preset, config), {})
