import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.flux2.flux2_text_to_image import Flux2TextToImage
from keras_hub.src.models.flux2.flux2_text_to_image_preprocessor import (
    Flux2TextToImagePreprocessor,
)
from keras_hub.src.models.flux2.flux2_vae import Flux2VAE
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.models.text_to_image import TextToImage
from keras_hub.src.tests.test_case import TestCase

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
]


class Flux2TextToImageTest(TestCase):
    def setUp(self):
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne", "u s", "e r", "us er"]
        merges += ["a s", "as s", "i s", "ass is", "t a", "n t", "ta nt"]
        merges += ["assis tant", "Ċ Ċ"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab = sorted(set(vocab)) + SPECIAL_TOKENS
        vocab = {token: i for i, token in enumerate(vocab)}
        tokenizer = Qwen3Tokenizer(
            vocabulary=vocab,
            merges=merges,
            unsplittable_tokens=SPECIAL_TOKENS,
        )
        self.preprocessor = Flux2TextToImagePreprocessor(
            tokenizer, sequence_length=24
        )
        self.text_encoder = Qwen3Backbone(
            vocabulary_size=len(vocab),
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            hidden_dim=16,
            intermediate_dim=32,
            name="text_encoder",
        )
        self.vae = Flux2VAE(
            [32, 32], [1, 1], [32, 32], [1, 1], latent_channels=4, name="vae"
        )
        self.backbone = self.make_backbone()
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
            # Two hidden states of width 16 give the backbone's 32 features.
            "text_encoder_layers": (1, 2),
        }
        self.input_data = {
            "latents": ops.ones((2, 8, 8, 16)),
            "token_ids": ops.ones((2, 24), dtype="int32"),
            "padding_mask": ops.ones((2, 24), dtype="int32"),
            "num_steps": ops.full((2,), 4, dtype="int32"),
        }

    def make_backbone(self, **kwargs):
        config = {
            "input_channels": 16,
            "hidden_size": 32,
            "mlp_ratio": 3.0,
            "num_heads": 2,
            "depth": 1,
            "depth_single_blocks": 1,
            "axes_dim": [4, 4, 4, 4],
            "theta": 2000,
            "latent_shape": (None, 16),
            "text_shape": (None, 32),
            "image_shape": (32, 32, 3),
        }
        config.update(kwargs)
        return Flux2Backbone(**config)

    def test_generate(self):
        text_to_image = Flux2TextToImage(**self.init_kwargs)
        self.assertEqual(text_to_image.latent_shape, (None, 8, 8, 16))
        self.assertFalse(text_to_image.support_negative_prompts)
        seed = 42
        image = text_to_image.generate("airplane", num_steps=2, seed=seed)
        self.assertEqual(image.shape, (32, 32, 3))
        self.assertEqual(image.dtype, np.uint8)
        # String input.
        prompts = ["airplane", "a user at the airport"]
        output = text_to_image.generate(prompts, num_steps=2, seed=seed)
        self.assertEqual(output.shape, (2, 32, 32, 3))
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess(prompts)
        text_to_image.preprocessor = None
        output2 = text_to_image.generate(
            {"prompts": prompt_ids}, num_steps=2, seed=seed
        )
        self.assertAllClose(output, output2)

    def test_distilled_model_ignores_guidance_scale(self):
        text_to_image = Flux2TextToImage(**self.init_kwargs)
        output = text_to_image.generate("airplane", num_steps=2, seed=42)
        output2 = text_to_image.generate(
            "airplane", num_steps=2, guidance_scale=7.0, seed=42
        )
        self.assertAllClose(output, output2)

    def test_generate_with_negative_prompts(self):
        text_to_image = Flux2TextToImage(**self.init_kwargs, is_distilled=False)
        self.assertTrue(text_to_image.support_negative_prompts)
        seed = 42
        prompts = ["airplane"]
        negative_prompts = ["a user"]
        # String input.
        output = text_to_image.generate(
            {"prompts": prompts, "negative_prompts": negative_prompts},
            num_steps=2,
            seed=seed,
        )
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess(prompts)
        negative_prompt_ids = self.preprocessor.generate_preprocess(
            negative_prompts
        )
        text_to_image.preprocessor = None
        output2 = text_to_image.generate(
            {"prompts": prompt_ids, "negative_prompts": negative_prompt_ids},
            num_steps=2,
            seed=seed,
        )
        self.assertAllClose(output, output2)

    def test_generate_compilation(self):
        if keras.config.backend() == "torch":
            self.skipTest("`generate()` is not compiled on the torch backend.")
        text_to_image = Flux2TextToImage(**self.init_kwargs)
        # The compiled `generate_step` only runs in Python while it is being
        # traced, so counting its calls counts traces.
        traces = []
        generate_step = text_to_image.generate_step

        def counting_generate_step(*args, **kwargs):
            traces.append(None)
            return generate_step(*args, **kwargs)

        text_to_image.generate_step = counting_generate_step
        # `num_steps` is a tensor and the schedule is computed per step, so
        # changing it must not retrace.
        text_to_image.generate("airplane", num_steps=2)
        text_to_image.generate("airplane", num_steps=3)
        self.assertLen(traces, 1)
        # `compile()` drops the compiled function, so the next call traces.
        text_to_image.compile()
        self.assertIsNone(text_to_image.generate_function)
        text_to_image.generate("airplane", num_steps=2)
        self.assertLen(traces, 2)

    def test_text_encoder_layers(self):
        # Indices follow Hugging Face `hidden_states`: `0` is the embedding
        # output, `i` the output of block `i - 1` and `num_layers` the
        # normalized final output. Features are concatenated in order.
        text_to_image = Flux2TextToImage(**self.init_kwargs)
        inputs = self.preprocessor.generate_preprocess(["airplane", "a user"])
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        embeddings = self.text_encoder.token_embedding(token_ids)
        block_0 = self.text_encoder.transformer_layers[0](
            embeddings, decoder_padding_mask=padding_mask
        )
        final = self.text_encoder(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )

        positive, negative = text_to_image.encode_text_step(inputs)
        self.assertIsNone(negative)
        self.assertAllClose(
            positive, ops.concatenate([block_0, final], axis=-1)
        )

        text_to_image = Flux2TextToImage(
            **{**self.init_kwargs, "text_encoder_layers": (2, 0)}
        )
        positive, _ = text_to_image.encode_text_step(inputs)
        self.assertAllClose(
            positive, ops.concatenate([final, embeddings], axis=-1)
        )

    def test_text_encoder_qk_norm_epsilon(self):
        # The Qwen3 query/key RMSNorms must use the checkpoint's
        # `rms_norm_eps`, like `Qwen3ForCausalLM`.
        Flux2TextToImage(**self.init_kwargs)
        for layer in self.text_encoder.transformer_layers:
            attention = layer._self_attention_layer
            for norm in (
                attention._query_dense_layer_norm,
                attention._key_dense_layer_norm,
            ):
                self.assertEqual(norm.epsilon, layer.layer_norm_epsilon)

    def test_encode_text_step_matches_transformers(self):
        """Every position, padding included, matches `Qwen3ForCausalLM`.

        FLUX.2 feeds all text positions to the transformer, so the padded
        ones matter too. The query/key projections are scaled down so their
        RMSNorm epsilon is not negligible, which is where the padded
        positions used to drift.
        """
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(0)
        hf_config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            rms_norm_eps=1e-6,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        hf_model = transformers.Qwen3ForCausalLM(hf_config).eval()
        with torch.no_grad():
            for layer in hf_model.model.layers:
                layer.self_attn.q_proj.weight.mul_(1e-2)
                layer.self_attn.k_proj.weight.mul_(1e-2)
        preset_dir = self.get_temp_dir()
        hf_model.save_pretrained(preset_dir)
        text_encoder = Qwen3Backbone.from_preset(preset_dir)

        # Right-padded, as the preprocessor pads.
        rng = np.random.default_rng(0)
        token_ids = rng.integers(1, 64, size=(2, 12)).astype("int32")
        padding_mask = np.zeros((2, 12), dtype="int32")
        padding_mask[0, :7] = 1
        padding_mask[1, :4] = 1
        token_ids = np.where(padding_mask == 1, token_ids, 0).astype("int32")

        text_to_image = Flux2TextToImage(
            **{**self.init_kwargs, "text_encoder": text_encoder}
        )
        positive, _ = text_to_image.encode_text_step(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )
        with torch.no_grad():
            hidden_states = hf_model(
                input_ids=torch.from_numpy(token_ids.astype("int64")),
                attention_mask=torch.from_numpy(padding_mask.astype("int64")),
                output_hidden_states=True,
            ).hidden_states
        expected = np.concatenate(
            [hidden_states[i].numpy() for i in (1, 2)], axis=-1
        )
        self.assertAllClose(positive, expected, atol=1e-5, rtol=1e-5)

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "guidance embedding"):
            Flux2TextToImage(
                **{
                    **self.init_kwargs,
                    "backbone": self.make_backbone(guidance_embed=True),
                }
            )
        with self.assertRaisesRegex(ValueError, "text_encoder_layers"):
            Flux2TextToImage(
                **{**self.init_kwargs, "text_encoder_layers": (1, 3)}
            )
        with self.assertRaisesRegex(ValueError, "text features"):
            Flux2TextToImage(
                **{**self.init_kwargs, "text_encoder_layers": (1,)}
            )
        with self.assertRaisesRegex(ValueError, "latent channels"):
            Flux2TextToImage(
                **{
                    **self.init_kwargs,
                    "backbone": self.make_backbone(
                        input_channels=32, latent_shape=(None, 32)
                    ),
                }
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            Flux2TextToImage(
                **{
                    **self.init_kwargs,
                    "backbone": self.make_backbone(image_shape=(30, 32, 3)),
                }
            )

    def test_serialization(self):
        self.run_serialization_test(Flux2TextToImage(**self.init_kwargs))

    def test_save_to_preset(self):
        text_to_image = Flux2TextToImage(**self.init_kwargs)
        output = text_to_image.generate("airplane", num_steps=2, seed=42)
        save_dir = self.get_temp_dir()
        text_to_image.save_to_preset(save_dir)
        restored = Flux2TextToImage.from_preset(save_dir)
        self.assertEqual(restored.text_encoder_layers, (1, 2))
        restored_output = restored.generate("airplane", num_steps=2, seed=42)
        self.assertAllClose(output, restored_output)

        # `from_preset(..., image_shape=...)` sets the generated image size,
        # also through the `TextToImage` base class. It does not touch the
        # backbone's latent token shape.
        for cls in (Flux2TextToImage, TextToImage):
            restored = cls.from_preset(save_dir, image_shape=(64, 32, 3))
            self.assertIsInstance(restored, Flux2TextToImage)
            self.assertEqual(restored.image_shape, (64, 32, 3))
            self.assertEqual(restored.latent_shape, (None, 16, 8, 16))
            self.assertEqual(
                tuple(restored.backbone.get_config()["latent_shape"]),
                (None, 16),
            )
            image = restored.generate("airplane", num_steps=1, seed=42)
            self.assertEqual(image.shape, (64, 32, 3))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Flux2TextToImage,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
