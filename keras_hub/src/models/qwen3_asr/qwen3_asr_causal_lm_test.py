from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm import Qwen3ASRCausalLM
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm_preprocessor import (
    Qwen3ASRCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.tensor_utils import tf


class Qwen3ASRCausalLMTest(TestCase):
    def setUp(self):
        keras.utils.set_random_seed(0)

        # Minimal BPE vocab
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = list("abcdefghijklmnopqrstuvwxyzĠĊ")
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])

        # Audio special tokens
        self.vocab += [
            "<|audio_pad|>",
            "<|audio_info|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<asr_text>",
        ]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])

        self.audio_token_id = self.vocab["<|audio_pad|>"]

        self.tokenizer = Qwen3ASRTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )

        self.audio_converter = Qwen3ASRAudioConverter(
            num_mels=128,
            sampling_rate=16000,
            n_window=50,
        )

        self.preprocessor = Qwen3ASRCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            audio_converter=self.audio_converter,
            sequence_length=40,
        )

        self.vocabulary_size = self.tokenizer.vocabulary_size()

        self.backbone = Qwen3ASRBackbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            hidden_dim=32,
            intermediate_dim=64,
            audio_num_mel_bins=128,
            audio_num_layers=2,
            audio_num_attention_heads=4,
            audio_intermediate_dim=64,
            audio_d_model=16,
            audio_n_window=50,
            audio_downsample_hidden_size=8,
            audio_max_position_embeddings=13,
            audio_token_id=self.audio_token_id,
        )

        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

        self.train_data = (
            {
                "prompts": [" airplane at airport", " airplane"],
                "audio": np.random.uniform(size=(2, 16000)),
                "responses": [" indeed", " yes"],
            },
        )
        # Preprocess to get input_data for saving test
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        # Grain (`_call_python` NumPy STFT) and `tf.data` (`_call_tf` STFT)
        # differ by ~1.3e-6, which TF32 matmuls on GPU amplify to ~2.5e-5.
        tol = 1e-4 if running_on_gpu() else 1e-6
        self.run_task_test(
            cls=Qwen3ASRCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 40, self.vocabulary_size),
            atol=tol,
            rtol=tol,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3ASRCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_generate(self):
        causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)

        prompt = " airplane"
        audio_input = np.random.uniform(size=(16000,)).astype("float32")
        inputs = {
            "prompts": prompt,
            "audio": audio_input,
        }

        preprocessed_prompt = self.preprocessor.generate_preprocess(
            {"prompts": [prompt], "audio": [audio_input]}
        )
        prompt_only = self.preprocessor.generate_postprocess(
            {
                "token_ids": preprocessed_prompt["token_ids"],
                "padding_mask": preprocessed_prompt["padding_mask"],
            }
        )[0]

        output = causal_lm.generate(inputs)
        self.assertIsInstance(output, str)
        self.assertIn(prompt, output)
        self.assertGreater(len(output), len(prompt_only))

        # Test batch generate
        batch_inputs = {
            "prompts": [" airplane", " airplane"],
            "audio": [
                np.random.uniform(size=(16000,)).astype("float32"),
                np.random.uniform(size=(8000,)).astype("float32"),
            ],
        }
        preprocessed_batch = self.preprocessor(batch_inputs)
        self.assertEqual(preprocessed_batch["audio_mel"].shape[0], 2)
        batch_output = causal_lm.generate(batch_inputs)
        self.assertEqual(len(batch_output), 2)
        self.assertIsInstance(batch_output[0], str)

    def test_generate_without_preprocessor_raises(self):
        causal_lm = Qwen3ASRCausalLM(backbone=self.backbone, preprocessor=None)
        with self.assertRaises(ValueError):
            causal_lm.generate(self.input_data)

    def test_generate_strip_prompt(self):
        causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)
        prompt = " airplane"
        inputs = {
            "prompts": prompt,
            "audio": np.zeros((16000,), dtype="float32"),
        }

        output = causal_lm.generate(inputs, strip_prompt=True)

        self.assertNotIn(prompt, output)

    def test_early_stopping(self):
        prompt = " airplane"
        inputs = {
            "prompts": [prompt],
            "audio": [np.zeros((16000,), dtype="float32")],
        }
        preprocessed_prompt = self.preprocessor.generate_preprocess(inputs)
        expected_output = self.preprocessor.generate_postprocess(
            {
                "token_ids": preprocessed_prompt["token_ids"],
                "padding_mask": preprocessed_prompt["padding_mask"],
            }
        )
        stop_token_id = self.tokenizer.end_token_id

        for stop_arg in ([stop_token_id], "auto"):
            causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)
            call_with_cache = causal_lm.call_with_cache

            def wrapper(*args, **kwargs):
                logits, hidden_states, cache = call_with_cache(*args, **kwargs)
                update = ops.ones_like(logits)[:, :, stop_token_id] * 1.0e9
                update = ops.expand_dims(update, axis=-1)
                logits = ops.slice_update(logits, (0, 0, stop_token_id), update)
                return logits, hidden_states, cache

            with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
                output = causal_lm.generate(
                    inputs,
                    stop_token_ids=stop_arg,
                )

            self.assertEqual(output, expected_output)

    def test_generate_compilation(self):
        causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)
        inputs = {
            "prompts": " airplane",
            "audio": np.zeros((16000,), dtype="float32"),
        }

        causal_lm.generate(inputs)
        first_fn = causal_lm.generate_function
        causal_lm.generate(inputs)
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)

        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_from_preset_round_trip(self):
        preset_dir = self.get_temp_dir()
        causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)
        causal_lm.save_to_preset(preset_dir)

        restored = Qwen3ASRCausalLM.from_preset(preset_dir)
        inputs = {
            "prompts": " airplane",
            "audio": np.zeros((16000,), dtype="float32"),
        }
        self.assertEqual(
            restored.generate(inputs, max_length=20),
            causal_lm.generate(inputs, max_length=20),
        )

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_generate_tf_dataset(self):
        causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "prompts": [" airplane", " airport"],
                "audio": np.zeros((2, 16000), dtype="float32"),
            }
        ).batch(2)
        outputs = causal_lm.generate(ds, max_length=25)
        self.assertEqual(len(outputs), 2)
        self.assertIsInstance(outputs[0], str)

    def test_generate_single_prompt_with_backend_tensor_audio(self):
        causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)
        audio_input = ops.convert_to_tensor(
            np.random.uniform(size=(16000,)).astype("float32")
        )

        output = causal_lm.generate(
            {"prompts": " airplane", "audio": audio_input}
        )

        self.assertIsInstance(output, str)
