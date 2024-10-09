import keras
import pytest
from keras import ops

from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (
    StableDiffusion3Backbone,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_inpaint import (
    StableDiffusion3Inpaint,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (
    StableDiffusion3TextToImagePreprocessor,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion3InpaintTest(TestCase):
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
            vae=VAEBackbone(
                [32, 32, 32, 32],
                [1, 1, 1, 1],
                [32, 32, 32, 32],
                [1, 1, 1, 1],
                # Use `mode` generate a deterministic output.
                sampler_method="mode",
                name="vae",
            ),
            clip_l=CLIPTextEncoder(
                20, 64, 64, 2, 2, 128, "quick_gelu", -2, name="clip_l"
            ),
            clip_g=CLIPTextEncoder(
                20, 128, 128, 2, 2, 256, "gelu", -2, name="clip_g"
            ),
            height=64,
            width=64,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.input_data = {
            "images": ops.ones((2, 64, 64, 3)),
            "latents": ops.ones((2, 8, 8, 16)),
            "clip_l_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_l_negative_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_g_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_g_negative_token_ids": ops.ones((2, 5), dtype="int32"),
            "num_steps": ops.ones((2,), dtype="int32"),
            "guidance_scale": ops.ones((2,)),
        }

    def test_inpaint_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=StableDiffusion3Inpaint,
            init_kwargs=self.init_kwargs,
            train_data=None,
            expected_output_shape={
                "images": (2, 64, 64, 3),
                "latents": (2, 8, 8, 16),
            },
        )

    def test_generate(self):
        inpaint = StableDiffusion3Inpaint(**self.init_kwargs)
        seed = 42
        image = self.input_data["images"][0]
        mask = self.input_data["images"][0][..., 0]  # (B, H, W)
        # String input.
        prompt = ["airplane"]
        negative_prompt = [""]
        output = inpaint.generate(
            {
                "images": image,
                "masks": mask,
                "prompts": prompt,
                "negative_prompts": negative_prompt,
            },
            seed=seed,
        )
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess(prompt)
        negative_prompt_ids = self.preprocessor.generate_preprocess(
            negative_prompt
        )
        inpaint.preprocessor = None
        output2 = inpaint.generate(
            {
                "images": image,
                "masks": mask,
                "prompts": prompt_ids,
                "negative_prompts": negative_prompt_ids,
            },
            seed=seed,
        )
        self.assertAllClose(output, output2)

    def test_generate_with_lower_precision(self):
        original_floatx = keras.config.floatx()
        try:
            for dtype in ["float16", "bfloat16"]:
                keras.config.set_floatx(dtype)
                inpaint = StableDiffusion3Inpaint(**self.init_kwargs)
                seed = 42
                image = self.input_data["images"][0]
                mask = self.input_data["images"][0][..., 0]  # (B, H, W)
                # String input.
                prompt = ["airplane"]
                negative_prompt = [""]
                output = inpaint.generate(
                    {
                        "images": image,
                        "masks": mask,
                        "prompts": prompt,
                        "negative_prompts": negative_prompt,
                    },
                    seed=seed,
                )
                # Int tensor input.
                prompt_ids = self.preprocessor.generate_preprocess(prompt)
                negative_prompt_ids = self.preprocessor.generate_preprocess(
                    negative_prompt
                )
                inpaint.preprocessor = None
                output2 = inpaint.generate(
                    {
                        "images": image,
                        "masks": mask,
                        "prompts": prompt_ids,
                        "negative_prompts": negative_prompt_ids,
                    },
                    seed=seed,
                )
                self.assertAllClose(output, output2)
        finally:
            # Restore floatx to the original value to prevent impact on other
            # tests even if there is an exception.
            keras.config.set_floatx(original_floatx)

    def test_generate_compilation(self):
        inpaint = StableDiffusion3Inpaint(**self.init_kwargs)
        image = self.input_data["images"][0]
        mask = self.input_data["images"][0][..., 0]  # (B, H, W)
        # Assert we do not recompile with successive calls.
        inpaint.generate(
            {
                "images": image,
                "masks": mask,
                "prompts": "airplane",
            }
        )
        first_fn = inpaint.generate_function
        inpaint.generate(
            {
                "images": image,
                "masks": mask,
                "prompts": "airplane",
            }
        )
        second_fn = inpaint.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        inpaint.compile()
        self.assertIsNone(inpaint.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=StableDiffusion3Inpaint,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
