from unittest.mock import patch

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_converter import (
    Qwen3OmniAudioConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
    Qwen3OmniAudioEncoder,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_causal_lm import (
    Qwen3OmniCausalLM,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_causal_lm_preprocessor import (
    Qwen3OmniCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_image_converter import (
    Qwen3OmniImageConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_tokenizer import (
    Qwen3OmniTokenizer,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_vision_encoder import (
    Qwen3OmniVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniCausalLMTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += [
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
        ]
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab = dict(
            [(token, i) for i, token in enumerate(dict.fromkeys(self.vocab))]
        )
        self.preprocessor = Qwen3OmniCausalLMPreprocessor(
            Qwen3OmniTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=7,
        )
        self.backbone = Qwen3OmniBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            moe_intermediate_dim=4,
            head_dim=2,
            num_experts=4,
            num_experts_per_tok=2,
            mrope_section=(1, 0, 0),
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen3OmniCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                7,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
        )

    def _build_multimodal_lm(
        self,
        image_converter=None,
        audio_converter=None,
        sequence_length=7,
    ):
        vision_encoder = Qwen3OmniVisionEncoder(
            depth=1,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=8,
            num_position_embeddings=49,
            deepstack_visual_indexes=[0],
        )
        audio_encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=8,
            max_source_positions=100,
            scale_embedding=False,
        )
        preprocessor = Qwen3OmniCausalLMPreprocessor(
            self.preprocessor.tokenizer,
            sequence_length=sequence_length,
            image_converter=image_converter,
            audio_converter=audio_converter,
        )
        backbone = Qwen3OmniBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            moe_intermediate_dim=4,
            head_dim=2,
            num_experts=4,
            num_experts_per_tok=2,
            mrope_section=(1, 0, 0),
            vision_encoder=vision_encoder,
            audio_encoder=audio_encoder,
        )
        causal_lm = Qwen3OmniCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        return causal_lm, preprocessor

    def test_multimodal_causal_lm_init(self):
        # `from_preset` builds a multimodal backbone inside the task,
        # so `__init__` must tolerate symbolic batch dims when filling
        # in default vision/audio inputs.
        causal_lm, preprocessor = self._build_multimodal_lm(
            image_converter=Qwen3OmniImageConverter(
                patch_size=2, temporal_patch_size=2
            ),
            audio_converter=Qwen3OmniAudioConverter(num_mels=80),
        )
        # Text-only prompts must still receive empty vision/audio
        # placeholders so the multimodal graph inputs are satisfied.
        pp = preprocessor.generate_preprocess(self.train_data[0])
        for key in (
            "pixel_values",
            "image_grid_thw",
            "vision_indices",
            "audio_features",
            "audio_indices",
        ):
            self.assertIn(key, pp)
        out = causal_lm(pp)
        self.assertEqual(
            out.shape[-1], self.preprocessor.tokenizer.vocabulary_size()
        )

    def test_multimodal_generate_preprocess_dict_call(self):
        # `generate_preprocess` emits precomputed `initial_embeddings`
        # (+ `deepstack_features`/`visual_pos_mask`) for the eager
        # encoder path. Feeding that dict straight into `causal_lm(...)`
        # must drop the extras rather than misalign the Functional
        # input-spec mapping (keras < 3.16 flattens dicts positionally).
        causal_lm, preprocessor = self._build_multimodal_lm(
            image_converter=Qwen3OmniImageConverter(
                patch_size=2,
                temporal_patch_size=2,
                min_pixels=32 * 32,
                max_pixels=64 * 64,
            ),
            sequence_length=200,
        )
        prompt = (
            "<|im_start|>user\n<|vision_start|><|image_pad|>"
            "<|vision_end|> airplane<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        image = np.random.default_rng(0).integers(
            0, 255, (32, 32, 3), dtype=np.uint8
        )
        pp = preprocessor.generate_preprocess(
            {"prompts": [prompt], "images": [image]}
        )
        self.assertIn("initial_embeddings", pp)
        self.assertIn("deepstack_features", pp)
        out = causal_lm(pp)
        self.assertEqual(
            out.shape[-1], self.preprocessor.tokenizer.vocabulary_size()
        )

    def test_generate(self):
        causal_lm = Qwen3OmniCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(" airplane at airport")
        self.assertTrue(prompt in output)
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            prompt_ids["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            prompt_ids["padding_mask"][:, :5],
        )

    def test_generate_strip_prompt(self):
        causal_lm = Qwen3OmniCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        causal_lm = Qwen3OmniCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = [" airplane at airport", " airplane"]
            output = causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = Qwen3OmniCausalLM(**self.init_kwargs)
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3OmniCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=Qwen3OmniCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_mrope_audio_branch(self):
        """Audio tokens advance all three channels in lock-step."""
        causal_lm = Qwen3OmniCausalLM(**self.init_kwargs)
        audio_token = 97
        audio_start = 96
        audio_end = 98
        # <pad><audio_start><audio_pad><audio_pad><audio_end><pad>
        token_ids = np.array(
            [[1, audio_start, audio_token, audio_token, audio_end, 2]],
            dtype=np.int64,
        )
        # _get_feat_extract_output_length(3) -> 2
        position_ids, deltas = causal_lm.compute_multimodal_rope_index(
            token_ids,
            audio_seqlens=np.array([3], dtype=np.int64),
            audio_token_id=audio_token,
            audio_start_token_id=audio_start,
        )
        self.assertEqual(position_ids.shape, (3, 1, 6))
        # All three channels match for the audio segment.
        self.assertAllEqual(position_ids[0, 0], position_ids[1, 0])
        self.assertAllEqual(position_ids[1, 0], position_ids[2, 0])

    def test_mrope_video_branch(self):
        """Video positions scale temporally with second_per_grids."""
        causal_lm = Qwen3OmniCausalLM(**self.init_kwargs)
        vision_start = 90
        vision_end = 91
        video_token = 93
        # <pad><vision_start><video><video><video><video><vision_end><pad>
        token_ids = np.array(
            [
                [
                    1,
                    vision_start,
                    video_token,
                    video_token,
                    video_token,
                    video_token,
                    vision_end,
                    2,
                ]
            ],
            dtype=np.int64,
        )
        # grid (T=1, H=4, W=4) => llm_h=llm_w=2 => 1*2*2 = 4 video tokens.
        video_grid_thw = np.array([[1, 4, 4]], dtype=np.int64)
        second_per_grids = np.array([1.0], dtype=np.float64)
        position_ids, _ = causal_lm.compute_multimodal_rope_index(
            token_ids,
            video_grid_thw=video_grid_thw,
            second_per_grids=second_per_grids,
            vision_start_token_id=vision_start,
            video_token_id=video_token,
        )
        self.assertEqual(position_ids.shape, (3, 1, 8))
        # Width channel should vary across the 4 video slots.
        vid_w = position_ids[2, 0, 2:6]
        self.assertGreater(int(vid_w.max() - vid_w.min()), 0)

    def test_mrope_audio_in_video_branch(self):
        """Audio-in-video merges audio and video positions in order."""
        causal_lm = Qwen3OmniCausalLM(**self.init_kwargs)
        vision_start = 90
        audio_start = 96
        audio_token = 97
        video_token = 93
        vision_end = 91
        audio_end = 98
        # <vision_start><audio_start><audio><audio><video><video>
        # <video><video><audio_end><vision_end>
        token_ids = np.array(
            [
                [
                    vision_start,
                    audio_start,
                    audio_token,
                    audio_token,
                    video_token,
                    video_token,
                    video_token,
                    video_token,
                    audio_end,
                    vision_end,
                ]
            ],
            dtype=np.int64,
        )
        video_grid_thw = np.array([[1, 4, 4]], dtype=np.int64)
        second_per_grids = np.array([1.0], dtype=np.float64)
        audio_seqlens = np.array([3], dtype=np.int64)
        position_ids, _ = causal_lm.compute_multimodal_rope_index(
            token_ids,
            video_grid_thw=video_grid_thw,
            second_per_grids=second_per_grids,
            audio_seqlens=audio_seqlens,
            use_audio_in_video=True,
            vision_start_token_id=vision_start,
            audio_start_token_id=audio_start,
            audio_token_id=audio_token,
            video_token_id=video_token,
        )
        self.assertEqual(position_ids.shape, (3, 1, 10))
        # Position IDs must be monotonically non-decreasing on the
        # temporal channel across the merged block.
        t_channel = position_ids[0, 0]
        self.assertTrue(bool(np.all(t_channel[1:] >= t_channel[:-1])))

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3OmniCausalLM.presets:
            self.run_preset_test(
                cls=Qwen3OmniCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
