from unittest.mock import patch

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_causal_lm import (
    UnlimitedOCRCausalLM,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_causal_lm_preprocessor import (  # noqa: E501
    UnlimitedOCRCausalLMPreprocessor,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_image_converter import (
    UnlimitedOCRImageConverter,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    OCR_MARKER_TOKENS,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    UnlimitedOCRTokenizer,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_vision_encoder import (
    UnlimitedOCRVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


class UnlimitedOCRCausalLMTest(TestCase):
    def setUp(self):
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["!", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]
        vocab += ["<｜▁pad▁｜>", "<image>"]
        vocab += list(OCR_MARKER_TOKENS)
        vocab = sorted(set(vocab))
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        tokenizer = UnlimitedOCRTokenizer(vocabulary=vocab, merges=merges)
        # A 64 pixel view gives 2x2 global tokens. A 40 pixel tile gives 2x2
        # local tokens.
        image_converter = UnlimitedOCRImageConverter(
            global_size=64,
            tile_size=40,
            min_tiles=2,
            max_tiles=6,
            patch_size=8,
            downsample_ratio=4,
        )
        self.preprocessor = UnlimitedOCRCausalLMPreprocessor(
            tokenizer, image_converter=image_converter, sequence_length=40
        )
        self.vocabulary_size = tokenizer.vocabulary_size()
        vision_encoder = UnlimitedOCRVisionEncoder(
            sam_hidden_dim=16,
            sam_num_layers=2,
            sam_num_heads=2,
            sam_mlp_dim=32,
            sam_output_channels=8,
            sam_patch_size=8,
            sam_image_size=128,
            sam_window_size=4,
            sam_global_attn_indexes=(1,),
            sam_downsample_channels=(12, 16),
            clip_hidden_dim=16,
            clip_num_layers=2,
            clip_num_heads=2,
            clip_intermediate_dim=32,
            clip_image_size=48,
            clip_patch_size=16,
        )
        self.backbone_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": 2,
            "num_query_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "moe_intermediate_dim": 8,
            "num_shared_experts": 2,
            "num_experts": 4,
            "top_k": 2,
            "vision_encoder": vision_encoder,
            "image_token_id": tokenizer.image_token_id,
            "global_image_size": 64,
            "local_image_size": 40,
            "sliding_window": 128,
        }
        self.backbone = UnlimitedOCRBackbone(**self.backbone_kwargs)
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        # Two 50x100 images. Each has two tiles and 17 image tokens.
        images = np.random.randint(0, 256, (2, 50, 100, 3)).astype("float32")
        self.train_data = (
            {
                "prompts": [
                    "<image>airplane at airport",
                    "<image>airplane",
                ],
                "images": images,
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=UnlimitedOCRCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 40, self.vocabulary_size),
        )

    def test_generate_with_images(self):
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        image = np.random.randint(0, 256, (30, 30, 3)).astype("float32")
        prompt = "<image>airplane at airport"
        output = causal_lm.generate({"prompts": [prompt], "images": [image]})
        # The preprocessor removes the image tokens. The text stays.
        self.assertIn("airplane at airport", output[0])

    def test_generate_text_only(self):
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        output = causal_lm.generate("airplane at airport")
        self.assertIn("airplane at airport", output)

    def test_generate_strip_prompt(self):
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_generate_matches_full_forward(self):
        # With a window longer than the sequence, the R-SWA equals full
        # causal attention. A greedy generation with the cache must then
        # pick the argmax of the full forward pass at each position. This
        # checks the image merge, the cache and the positions.
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        causal_lm.compile(sampler="greedy")
        # Large logits keep the argmax stable across the two code paths.
        # The model must not generate the image token. A full forward pass
        # would merge image features into it, but a decode step would not.
        embedding = causal_lm.backbone.token_embedding.reverse_embeddings
        values = np.random.normal(scale=3.0, size=embedding.shape)
        values[:, self.preprocessor.tokenizer.image_token_id] = 0.0
        embedding.assign(values.astype("float32"))
        image = np.random.randint(0, 256, (30, 30, 3)).astype("float32")
        inputs = self.preprocessor.generate_preprocess(
            {"prompts": ["<image>airplane"], "images": [image]}
        )
        padding_mask = ops.convert_to_numpy(inputs["padding_mask"])
        prompt_length = int(np.sum(padding_mask))
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(inputs, stop_token_ids=None)
        token_ids = ops.convert_to_numpy(outputs["token_ids"])
        full_inputs = dict(inputs)
        full_inputs["token_ids"] = ops.convert_to_tensor(token_ids)
        full_inputs["padding_mask"] = ops.ones_like(inputs["padding_mask"])
        logits = ops.convert_to_numpy(causal_lm(full_inputs))
        predictions = logits.argmax(axis=-1)
        generated = np.arange(prompt_length - 1, token_ids.shape[1] - 1)
        self.assertAllEqual(
            predictions[0, generated], token_ids[0, generated + 1]
        )

    def test_ragged_generate_matches_reference_window_mask(self):
        # The window (3) is shorter than the decode length. The prompts have
        # different lengths, so each row has its own prefill length. The
        # generated tokens must equal the argmax of a full forward pass that
        # uses an explicit R-SWA mask with the prefill length of the row.
        window = 3
        tokenizer = self.preprocessor.tokenizer
        backbone = UnlimitedOCRBackbone(
            **{**self.backbone_kwargs, "sliding_window": window}
        )
        causal_lm = UnlimitedOCRCausalLM(
            backbone=backbone, preprocessor=self.preprocessor
        )
        causal_lm.compile(sampler="greedy")
        # Large logits keep the argmax stable across the two code paths.
        # The image token gets a zero logit, so generation never picks it.
        embedding = backbone.token_embedding.reverse_embeddings
        values = np.random.normal(scale=3.0, size=embedding.shape)
        values[:, tokenizer.image_token_id] = 0.0
        embedding.assign(values.astype("float32"))

        inputs = self.preprocessor.generate_preprocess(
            ["airplane at airport", "airplane"]
        )
        prompt_lengths = ops.convert_to_numpy(inputs["padding_mask"]).sum(
            axis=-1
        )
        self.assertNotEqual(prompt_lengths[0], prompt_lengths[1])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(inputs, stop_token_ids=None)
        token_ids = ops.convert_to_numpy(outputs["token_ids"])
        sequence_length = token_ids.shape[1]
        self.assertGreater(sequence_length - prompt_lengths.min(), window)

        query = np.arange(sequence_length)[:, None]
        key = np.arange(sequence_length)[None, :]

        def predictions(masks):
            x = backbone.token_embedding(token_ids)
            for layer in backbone.transformer_layers:
                x = layer(x, decoder_attention_mask=np.stack(masks))
            x = backbone.layer_norm(x)
            return ops.convert_to_numpy(
                backbone.token_embedding(x, reverse=True)
            )

        reference_masks = [
            (key <= query) & ((key < length) | (key > query - window))
            for length in prompt_lengths
        ]
        logits = predictions(reference_masks)
        for row, length in enumerate(prompt_lengths):
            positions = np.arange(length - 1, sequence_length - 1)
            self.assertAllEqual(
                logits[row].argmax(axis=-1)[positions],
                token_ids[row, positions + 1],
            )

        # Guard against a window that never applies: full causal attention
        # must give other logits at the decoded positions.
        causal_logits = predictions([key <= query] * 2)
        difference = np.abs(logits[:, -1] - causal_logits[:, -1]).max()
        self.assertGreater(float(difference), 1e-3)

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_text_only_predict_evaluate_and_fit(self):
        # A text-only batch has image inputs with a batch size of 0. They go
        # through the compiled `predict`, `evaluate` and `fit` functions.
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        prompts = ["airplane at airport", "airplane"]
        ds = tf.data.Dataset.from_tensor_slices(prompts).batch(2)
        output = causal_lm.predict(prompts)
        self.assertEqual(output.shape, (2, 40, self.vocabulary_size))
        self.assertAllClose(causal_lm.predict(ds), output)
        causal_lm.evaluate(ds)
        causal_lm.fit(ds)
        # With split preprocessing.
        preprocessor = causal_lm.preprocessor
        causal_lm.preprocessor = None
        split_ds = ds.map(preprocessor)
        causal_lm.predict(split_ds)
        causal_lm.evaluate(split_ds)
        causal_lm.fit(split_ds)
        causal_lm.preprocessor = preprocessor

    def test_decode_steps_get_the_prompt_length(self):
        # The prefill pass has no prompt length. Every decode step gets the
        # number of prompt tokens of each row for the R-SWA mask.
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache
        calls = []

        def wrapper(*args, **kwargs):
            calls.append(kwargs.get("prefill_length"))
            return call_with_cache(*args, **kwargs)

        image = np.random.randint(0, 256, (30, 30, 3)).astype("float32")
        inputs = self.preprocessor.generate_preprocess(
            {"prompts": ["<image>airplane"], "images": [image]}
        )
        padding_mask = ops.convert_to_numpy(inputs["padding_mask"])
        prompt_length = int(np.sum(padding_mask))
        causal_lm.preprocessor = None
        causal_lm.run_eagerly = True
        causal_lm.generate_function = None
        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            causal_lm.generate(inputs, stop_token_ids=None)
        self.assertIsNone(calls[0])
        self.assertGreater(len(calls), 1)
        for prefill_length in calls[1:]:
            self.assertAllEqual(prefill_length, [prompt_length])

    def test_early_stopping(self):
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["airplane at airport", "airplane"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate("airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_quantize(self):
        causal_lm = UnlimitedOCRCausalLM(**self.init_kwargs)
        causal_lm.backbone(self.input_data)
        causal_lm.quantize("int8")
        output = causal_lm.generate("airplane at airport")
        self.assertIn("airplane at airport", output)
        logits = causal_lm(self.input_data)
        self.assertEqual(logits.shape, (2, 40, self.vocabulary_size))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=UnlimitedOCRCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in UnlimitedOCRCausalLM.presets:
            self.run_preset_test(
                cls=UnlimitedOCRCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
