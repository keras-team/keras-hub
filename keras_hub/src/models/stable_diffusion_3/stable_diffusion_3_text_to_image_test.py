# Copyright 2024 The KerasHub Authors
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
import keras
import pytest
from keras import ops

from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (
    StableDiffusion3Backbone,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image import (
    StableDiffusion3TextToImage,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (
    StableDiffusion3TextToImagePreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion3TextToImageTest(TestCase):
    def setUp(self):
        # Instantiate the preprocessor.
        vocab = ["air", "plane</w>", "port</w>"]
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        clip_l_tokenizer = CLIPTokenizer(vocab, merges, pad_with_end_token=True)
        clip_g_tokenizer = CLIPTokenizer(vocab, merges)
        clip_l_preprocessor = CLIPPreprocessor(clip_l_tokenizer)
        clip_g_preprocessor = CLIPPreprocessor(clip_g_tokenizer)
        self.preprocessor = StableDiffusion3TextToImagePreprocessor(
            clip_l_preprocessor, clip_g_preprocessor
        )

        self.backbone = StableDiffusion3Backbone(
            mmdit_patch_size=2,
            mmdit_hidden_dim=16 * 2,
            mmdit_num_layers=2,
            mmdit_num_heads=2,
            mmdit_position_size=192,
            vae_stackwise_num_filters=[32, 32, 32, 32],
            vae_stackwise_num_blocks=[1, 1, 1, 1],
            clip_l=CLIPTextEncoder(
                20, 64, 64, 2, 2, 128, "quick_gelu", -2, name="clip_l"
            ),
            clip_g=CLIPTextEncoder(
                20, 128, 128, 2, 2, 256, "gelu", -2, name="clip_g"
            ),
            height=128,
            width=128,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.input_data = {
            "latents": ops.ones((2, 16, 16, 16)),
            "clip_l_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_l_negative_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_g_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_g_negative_token_ids": ops.ones((2, 5), dtype="int32"),
            "num_steps": ops.ones((2,), dtype="int32"),
            "guidance_scale": ops.ones((2,)),
        }

    def test_text_to_image_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=StableDiffusion3TextToImage,
            init_kwargs=self.init_kwargs,
            train_data=None,
            expected_output_shape=(2, 128, 128, 3),
        )

    def test_generate(self):
        text_to_image = StableDiffusion3TextToImage(**self.init_kwargs)
        seed = 42
        # String input.
        prompt = ["airplane"]
        negative_prompt = [""]
        output = text_to_image.generate(prompt, negative_prompt, seed=seed)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess(prompt)
        negative_prompt_ids = self.preprocessor.generate_preprocess(
            negative_prompt
        )
        text_to_image.preprocessor = None
        output2 = text_to_image.generate(
            prompt_ids, negative_prompt_ids, seed=seed
        )
        self.assertAllClose(output, output2)

    def test_generate_with_lower_precision(self):
        original_floatx = keras.config.floatx()
        try:
            for dtype in ["float16", "bfloat16"]:
                keras.config.set_floatx(dtype)
                text_to_image = StableDiffusion3TextToImage(**self.init_kwargs)
                seed = 42
                # String input.
                prompt = ["airplane"]
                negative_prompt = [""]
                output = text_to_image.generate(
                    prompt, negative_prompt, seed=seed
                )
                # Int tensor input.
                prompt_ids = self.preprocessor.generate_preprocess(prompt)
                negative_prompt_ids = self.preprocessor.generate_preprocess(
                    negative_prompt
                )
                text_to_image.preprocessor = None
                output2 = text_to_image.generate(
                    prompt_ids, negative_prompt_ids, seed=seed
                )
                self.assertAllClose(output, output2)
        finally:
            # Restore floatx to the original value to prevent impact on other
            # tests even if there is an exception.
            keras.config.set_floatx(original_floatx)

    def test_generate_compilation(self):
        text_to_image = StableDiffusion3TextToImage(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        text_to_image.generate("airplane")
        first_fn = text_to_image.generate_function
        text_to_image.generate("airplane")
        second_fn = text_to_image.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        text_to_image.compile()
        self.assertIsNone(text_to_image.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=StableDiffusion3TextToImage,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
