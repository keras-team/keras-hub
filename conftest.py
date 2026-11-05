import os

import keras
import pytest

# OpenVINO supported test paths
OPENVINO_SUPPORTED_PATHS = [
    "integration_tests",
    "keras_hub/src/layers/modeling/alibi_bias_test.py",
    "keras_hub/src/layers/modeling/anchor_generator_test.py",
    "keras_hub/src/layers/modeling/box_matcher_test.py",
    "keras_hub/src/layers/modeling/f_net_encoder_test.py",
    "keras_hub/src/layers/modeling/masked_lm_head_test.py",
    "keras_hub/src/layers/modeling/non_max_supression_test.py",
    "keras_hub/src/layers/modeling/position_embedding_test.py",
    "keras_hub/src/layers/modeling/rotary_embedding_test.py",
    "keras_hub/src/layers/modeling/sine_position_encoding_test.py",
    "keras_hub/src/layers/modeling/token_and_position_embedding_test.py",
    "keras_hub/src/layers/modeling/transformer_encoder_test.py",
    "keras_hub/src/layers/modeling/transformer_layer_utils_test.py",
    "keras_hub/src/layers/preprocessing/audio_converter_test.py",
    "keras_hub/src/layers/preprocessing/masked_lm_mask_generator_test.py",
    "keras_hub/src/layers/preprocessing/multi_segment_packer_test.py",
    "keras_hub/src/layers/preprocessing/random_deletion_test.py",
    "keras_hub/src/layers/preprocessing/random_swap_test.py",
    "keras_hub/src/layers/preprocessing/start_end_packer_test.py",
    "keras_hub/src/metrics",
    "keras_hub/src/samplers/serialization_test.py",
    "keras_hub/src/tests",
    "keras_hub/src/tokenizers",
    "keras_hub/src/utils/audio_utils_test.py",
    "keras_hub/src/utils/coco",
    "keras_hub/src/utils/imagenet",
    "keras_hub/src/utils/keras_utils_test.py",
    "keras_hub/src/utils/openvino_utils_test.py",
    "keras_hub/src/utils/pipeline_model_test.py",
    "keras_hub/src/utils/preset_utils_test.py",
    "keras_hub/src/utils/python_utils_test.py",
    "keras_hub/src/utils/random_utils_test.py",
    "keras_hub/src/utils/tensor_utils_test.py",
    "keras_hub/src/utils/timm",
    "keras_hub/src/utils/transformers/convert_albert_test.py",
    "keras_hub/src/utils/transformers/convert_bart_test.py",
    "keras_hub/src/utils/transformers/convert_bert_test.py",
    "keras_hub/src/utils/transformers/convert_blip2_test.py",
    "keras_hub/src/utils/transformers/convert_deit_test.py",
    "keras_hub/src/utils/transformers/convert_diffusion_gemma_test.py",
    "keras_hub/src/utils/transformers/convert_dinov2_test.py",
    "keras_hub/src/utils/transformers/convert_dinov3_test.py",
    "keras_hub/src/utils/transformers/convert_distilbert_test.py",
    "keras_hub/src/utils/transformers/convert_embedding_gemma2_test.py",
    "keras_hub/src/utils/transformers/convert_esm_test.py",
    "keras_hub/src/utils/transformers/convert_gemma3_test.py",
    "keras_hub/src/utils/transformers/convert_gemma3n_test.py",
    "keras_hub/src/utils/transformers/convert_gemma4_assistant_test.py",
    "keras_hub/src/utils/transformers/convert_gemma_test.py",
    "keras_hub/src/utils/transformers/convert_gpt2_test.py",
    "keras_hub/src/utils/transformers/convert_gpt_oss_test.py",
    "keras_hub/src/utils/transformers/convert_llama3_test.py",
    "keras_hub/src/utils/transformers/convert_metaclip_2_test.py",
    "keras_hub/src/utils/transformers/convert_mistral3_test.py",
    "keras_hub/src/utils/transformers/convert_mistral_test.py",
    "keras_hub/src/utils/transformers/convert_mixtral_test.py",
    "keras_hub/src/utils/transformers/convert_modern_bert_test.py",
    "keras_hub/src/utils/transformers/convert_mpnet_test.py",
    "keras_hub/src/utils/transformers/convert_muse_glimmer_assistant_test.py",
    "keras_hub/src/utils/transformers/convert_muse_glimmer_test.py",
    "keras_hub/src/utils/transformers/convert_pali_gemma_test.py",
    "keras_hub/src/utils/transformers/convert_qwen3_5_moe_test.py",
    "keras_hub/src/utils/transformers/convert_qwen3_5_test.py",
    "keras_hub/src/utils/transformers/convert_qwen3_moe_test.py",
    "keras_hub/src/utils/transformers/convert_qwen3_test.py",
    "keras_hub/src/utils/transformers/convert_qwen_moe_test.py",
    "keras_hub/src/utils/transformers/convert_qwen_test.py",
    "keras_hub/src/utils/transformers/convert_roberta_test.py",
    "keras_hub/src/utils/transformers/convert_sam3_test.py",
    "keras_hub/src/utils/transformers/convert_smollm3_test.py",
    "keras_hub/src/utils/transformers/convert_smolvlm2_test.py",
    "keras_hub/src/utils/transformers/convert_swin_transformer_test.py",
    "keras_hub/src/utils/transformers/convert_t5gemma2_test.py",
    "keras_hub/src/utils/transformers/convert_t5gemma_test.py",
    "keras_hub/src/utils/transformers/convert_vit_test.py",
    "keras_hub/src/utils/transformers/convert_xlm_roberta_test.py",
    "keras_hub/src/utils/transformers/export/gemma_test.py",
    "keras_hub/src/utils/transformers/export/gpt2_test.py",
    "keras_hub/src/utils/transformers/export/llama3_test.py",
    "keras_hub/src/utils/transformers/export/mistral_test.py",
    "keras_hub/src/utils/transformers/export/qwen3_5_test.py",
    "keras_hub/src/utils/transformers/export/qwen_test.py",
    "keras_hub/src/vllm",
    "keras_hub/src/models/albert",
    "keras_hub/src/models/audio_to_text_preprocessor_test.py",
    "keras_hub/src/models/backbone_test.py",
    "keras_hub/src/models/bart",
    "keras_hub/src/models/basnet",
    "keras_hub/src/models/bert",
    "keras_hub/src/models/blip2/blip2_backbone_test.py",
    "keras_hub/src/models/blip2/blip2_causal_lm_preprocessor_test.py",
    "keras_hub/src/models/blip2/blip2_causal_lm_test.py",
    "keras_hub/src/models/blip2/blip2_custom_opt_test.py",
    "keras_hub/src/models/blip2/blip2_flan_t5_lm_test.py",
    "keras_hub/src/models/blip2/blip2_flan_t5_tokenizer_test.py",
    "keras_hub/src/models/blip2/blip2_image_converter_test.py",
    "keras_hub/src/models/blip2/blip2_opt_tokenizer_test.py",
    "keras_hub/src/models/blip2/blip2_seq_2_seq_lm_test.py",
    "keras_hub/src/models/blip2/blip2_vision_encoder_test.py",
    "keras_hub/src/models/block_diffusion_lm_preprocessor_test.py",
    "keras_hub/src/models/bloom",
    "keras_hub/src/models/causal_lm_preprocessor_test.py",
    "keras_hub/src/models/clip/clip_preprocessor_test.py",
    "keras_hub/src/models/clip/clip_tokenizer_test.py",
    "keras_hub/src/models/cspnet",
    "keras_hub/src/models/deberta_v3",
    "keras_hub/src/models/deeplab_v3",
    "keras_hub/src/models/deit",
    "keras_hub/src/models/densenet",
    "keras_hub/src/models/depth_anything",
    "keras_hub/src/models/diffusion_gemma/diffusion_gemma_backbone_test.py",
    "keras_hub/src/models/diffusion_gemma/diffusion_gemma_block_diffusion_lm_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/diffusion_gemma/diffusion_gemma_layers_test.py",
    "keras_hub/src/models/diffusion_gemma/diffusion_gemma_transformer_layer_test.py",  # noqa: E501
    "keras_hub/src/models/dinov2",
    "keras_hub/src/models/dinov3",
    "keras_hub/src/models/distil_bert",
    "keras_hub/src/models/edrec",
    "keras_hub/src/models/efficientnet",
    "keras_hub/src/models/electra",
    "keras_hub/src/models/embedding_gemma2/embedding_gemma2_audio_converter_test.py",  # noqa: E501
    "keras_hub/src/models/embedding_gemma2/embedding_gemma2_backbone_test.py",
    "keras_hub/src/models/embedding_gemma2/embedding_gemma2_encoder_block_test.py",  # noqa: E501
    "keras_hub/src/models/embedding_gemma2/embedding_gemma2_image_converter_test.py",  # noqa: E501
    "keras_hub/src/models/embedding_gemma2/embedding_gemma2_lora_test.py",
    "keras_hub/src/models/embedding_gemma2/embedding_gemma2_tokenizer_test.py",
    "keras_hub/src/models/embedding_gemma2/embedding_gemma2_video_converter_test.py",  # noqa: E501
    "keras_hub/src/models/esm/esm_classifier_preprocessor_test.py",
    "keras_hub/src/models/esm/esm_classifier_test.py",
    "keras_hub/src/models/esm/esm_masked_plm_preprocessor_test.py",
    "keras_hub/src/models/esm/esm_masked_plm_test.py",
    "keras_hub/src/models/esm/esm_tokenizer_test.py",
    "keras_hub/src/models/f_net",
    "keras_hub/src/models/falcon",
    "keras_hub/src/models/flux",
    "keras_hub/src/models/gemma",
    "keras_hub/src/models/gemma3",
    "keras_hub/src/models/gemma3n",
    "keras_hub/src/models/gemma4/gemma4_audio_converter_test.py",
    "keras_hub/src/models/gemma4/gemma4_backbone_test.py",
    "keras_hub/src/models/gemma4/gemma4_decoder_block_test.py",
    "keras_hub/src/models/gemma4/gemma4_image_converter_test.py",
    "keras_hub/src/models/gemma4/gemma4_layers_test.py",
    "keras_hub/src/models/gemma4/gemma4_lora_test.py",
    "keras_hub/src/models/gemma4/gemma4_tokenizer_test.py",
    "keras_hub/src/models/gemma4/gemma4_video_converter_test.py",
    "keras_hub/src/models/gemma4/gemma4_vision_encoder_test.py",
    "keras_hub/src/models/gpt2",
    "keras_hub/src/models/gpt_neo_x/gpt_neo_x_causal_lm_preprocessor_test.py",
    "keras_hub/src/models/gpt_neo_x/gpt_neo_x_tokenizer_test.py",
    "keras_hub/src/models/gpt_oss",
    "keras_hub/src/models/hgnetv2",
    "keras_hub/src/models/llama",
    "keras_hub/src/models/llama3",
    "keras_hub/src/models/masked_lm_preprocessor_test.py",
    "keras_hub/src/models/metaclip_2/metaclip_2_preprocessor_test.py",
    "keras_hub/src/models/metaclip_2/metaclip_2_tokenizer_test.py",
    "keras_hub/src/models/mistral",
    "keras_hub/src/models/mistral3",
    "keras_hub/src/models/mit",
    "keras_hub/src/models/mixtral",
    "keras_hub/src/models/mobilenet",
    "keras_hub/src/models/mobilenetv5",
    "keras_hub/src/models/modern_bert/modern_bert_backbone_test.py",
    "keras_hub/src/models/modern_bert/modern_bert_layers_test.py",
    "keras_hub/src/models/modern_bert/modern_bert_masked_lm_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/modern_bert/modern_bert_masked_lm_test.py",
    "keras_hub/src/models/modern_bert/modern_bert_text_classifier_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/modern_bert/modern_bert_text_classifier_test.py",
    "keras_hub/src/models/modern_bert/modern_bert_text_embedder_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/modern_bert/modern_bert_tokenizer_test.py",
    "keras_hub/src/models/moonshine/moonshine_audio_converter_test.py",
    "keras_hub/src/models/moonshine/moonshine_audio_to_text_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/moonshine/moonshine_backbone_test.py",
    "keras_hub/src/models/moonshine/moonshine_layers_test.py",
    "keras_hub/src/models/moonshine/moonshine_tokenizer_test.py",
    "keras_hub/src/models/mpnet",
    "keras_hub/src/models/muse_glimmer/muse_glimmer_assistant_causal_lm_test.py",  # noqa: E501
    "keras_hub/src/models/muse_glimmer/muse_glimmer_attention_test.py",
    "keras_hub/src/models/muse_glimmer/muse_glimmer_causal_lm_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/muse_glimmer/muse_glimmer_decoder_test.py",
    "keras_hub/src/models/muse_glimmer/muse_glimmer_layers_test.py",
    "keras_hub/src/models/muse_glimmer/muse_glimmer_tokenizer_test.py",
    "keras_hub/src/models/muse_glimmer/muse_glimmer_video_converter_test.py",
    "keras_hub/src/models/muse_glimmer/muse_glimmer_vision_encoder_test.py",
    "keras_hub/src/models/opt",
    "keras_hub/src/models/pali_gemma",
    "keras_hub/src/models/parseq",
    "keras_hub/src/models/phi3",
    "keras_hub/src/models/preprocessor_test.py",
    "keras_hub/src/models/qwen",
    "keras_hub/src/models/qwen3",
    "keras_hub/src/models/qwen3_5/qwen3_5_backbone_test.py",
    "keras_hub/src/models/qwen3_5/qwen3_5_video_converter_test.py",
    "keras_hub/src/models/qwen3_5/qwen3_5_vision_encoder_test.py",
    "keras_hub/src/models/qwen3_5_moe/qwen3_5_moe_backbone_test.py",
    "keras_hub/src/models/qwen3_5_moe/qwen3_5_moe_causal_lm_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/qwen3_moe",
    "keras_hub/src/models/qwen_moe",
    "keras_hub/src/models/resnet/resnet_image_classifier_test.py",
    "keras_hub/src/models/retinanet",
    "keras_hub/src/models/roberta",
    "keras_hub/src/models/roformer_v2/roformer_v2_masked_lm_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/roformer_v2/roformer_v2_text_classifier_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/roformer_v2/roformer_v2_tokenizer_test.py",
    "keras_hub/src/models/rqvae",
    "keras_hub/src/models/rwkv7/rwkv7_causal_lm_preprocessor_test.py",
    "keras_hub/src/models/rwkv7/rwkv7_tokenizer_test.py",
    "keras_hub/src/models/sam/sam_image_segmenter_test.py",
    "keras_hub/src/models/sam/sam_mask_decoder_test.py",
    "keras_hub/src/models/sam/sam_prompt_encoder_test.py",
    "keras_hub/src/models/segformer",
    "keras_hub/src/models/seq_2_seq_lm_preprocessor_test.py",
    "keras_hub/src/models/siglip",
    "keras_hub/src/models/smollm3",
    "keras_hub/src/models/smolvlm2/smolvlm2_backbone_test.py",
    "keras_hub/src/models/smolvlm2/smolvlm2_causal_lm_preprocessor_test.py",
    "keras_hub/src/models/smolvlm2/smolvlm2_causal_lm_test.py",
    "keras_hub/src/models/smolvlm2/smolvlm2_tokenizer_test.py",
    "keras_hub/src/models/smolvlm2/smolvlm2_video_converter_test.py",
    "keras_hub/src/models/smolvlm2/smolvlm2_vision_encoder_test.py",
    "keras_hub/src/models/stable_diffusion_3/stable_diffusion_3_image_to_image_test.py",  # noqa: E501
    "keras_hub/src/models/stable_diffusion_3/stable_diffusion_3_inpaint_test.py",  # noqa: E501
    "keras_hub/src/models/stable_diffusion_3/stable_diffusion_3_text_to_image_preprocessor_test.py",  # noqa: E501
    "keras_hub/src/models/swin_transformer",
    "keras_hub/src/models/t5",
    "keras_hub/src/models/t5gemma",
    "keras_hub/src/models/t5gemma2",
    "keras_hub/src/models/task_test.py",
    "keras_hub/src/models/text_classifier_preprocessor_test.py",
    "keras_hub/src/models/text_embedder_preprocessor_test.py",
    "keras_hub/src/models/text_to_image_preprocessor_test.py",
    "keras_hub/src/models/vae",
    "keras_hub/src/models/vgg",
    "keras_hub/src/models/video_prism",
    "keras_hub/src/models/vit",
    "keras_hub/src/models/vit_det",
    "keras_hub/src/models/whisper",
    "keras_hub/src/models/xception",
    "keras_hub/src/models/xlm_roberta",
    "keras_hub/src/models/xlnet",
]


def pytest_addoption(parser):
    parser.addoption(
        "--run_large",
        action="store_true",
        default=False,
        help="run large tests",
    )
    parser.addoption(
        "--run_extra_large",
        action="store_true",
        default=False,
        help="run extra_large tests",
    )
    parser.addoption(
        "--docstring_module",
        action="store",
        default="",
        help="restrict docs testing to modules whose name matches this flag",
    )
    parser.addoption(
        "--check_gpu",
        action="store_true",
        default=False,
        help="fail if a gpu is not present",
    )


def pytest_configure(config):
    # Monkey-patch training methods for OpenVINO backend
    if keras.config.backend() == "openvino":
        keras.Model.fit = lambda *args, **kwargs: pytest.skip(
            "Model.fit() not supported on OpenVINO backend"
        )
        keras.Model.train_on_batch = lambda *args, **kwargs: pytest.skip(
            "Model.train_on_batch() not supported on OpenVINO backend"
        )

    # Verify that device has GPU and detected by backend
    if config.getoption("--check_gpu"):
        found_gpu = False
        backend = keras.config.backend()
        if backend == "jax":
            import jax

            try:
                found_gpu = bool(jax.devices("gpu"))
            except RuntimeError:
                found_gpu = False
        elif backend == "tensorflow":
            import tensorflow as tf

            found_gpu = bool(tf.config.list_logical_devices("GPU"))
        elif backend == "torch":
            import torch

            found_gpu = bool(torch.cuda.device_count())
        if not found_gpu:
            pytest.fail(f"No GPUs discovered on the {backend} backend.")

    config.addinivalue_line(
        "markers",
        "large: mark test as being slow or requiring a network",
    )
    config.addinivalue_line(
        "markers",
        "extra_large: mark test as being too large to run continuously",
    )
    config.addinivalue_line(
        "markers",
        "tf_only: mark test as a tf only test",
    )
    config.addinivalue_line(
        "markers",
        "kaggle_key_required: mark test needing a kaggle key",
    )


def pytest_collection_modifyitems(config, items):
    run_extra_large_tests = config.getoption("--run_extra_large")
    run_large_tests = config.getoption("--run_large")

    # Messages to annotate skipped tests with.
    skip_large = pytest.mark.skipif(
        not run_large_tests,
        reason="need --run_large option to run",
    )
    skip_extra_large = pytest.mark.skipif(
        not run_extra_large_tests,
        reason="need --run_extra_large option to run",
    )
    tf_only = pytest.mark.skipif(
        not keras.config.backend() == "tensorflow",
        reason="tests only run on tf backend",
    )
    found_kaggle_key = all(
        [
            os.environ.get("KAGGLE_USERNAME", None),
            os.environ.get("KAGGLE_KEY", None),
        ]
    )
    kaggle_key_required = pytest.mark.skipif(
        not found_kaggle_key,
        reason="tests only run with a kaggle api key",
    )
    for item in items:
        if "large" in item.keywords and "extra_large" not in item.keywords:
            item.add_marker(skip_large)
        if "extra_large" in item.keywords:
            item.add_marker(skip_extra_large)
        if "tf_only" in item.keywords:
            item.add_marker(tf_only)
        if "kaggle_key_required" in item.keywords:
            item.add_marker(kaggle_key_required)

        # OpenVINO-specific test skipping
        if keras.config.backend() == "openvino":
            is_whitelisted = any(
                item.nodeid.startswith(supported_path + "/")
                or item.nodeid.startswith(supported_path + "::")
                or item.nodeid == supported_path
                for supported_path in OPENVINO_SUPPORTED_PATHS
            )

            if not is_whitelisted:
                item.add_marker(
                    pytest.mark.skipif(
                        True, reason="OpenVINO: File/directory not in whitelist"
                    )
                )


# Disable traceback filtering for quicker debugging of tests failures.
keras.config.disable_traceback_filtering()
