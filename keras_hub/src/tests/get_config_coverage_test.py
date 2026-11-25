import ast
import importlib
import inspect
import pathlib
import pkgutil
import textwrap

import keras
import pytest

from keras_hub.src.models.task import Task
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.tokenizer import Tokenizer

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]

# `self`, and the arguments that the base Keras classes serialize.
SKIPPED_ARGUMENTS = ("self", "name", "dtype", "trainable")

# Constructor arguments that are left out of `get_config()` on purpose, as
# `{(class, argument): reason}`. An entry also covers the subclasses.
INTENTIONAL = {
    (Tokenizer, "vocabulary"): "Saved as an asset file by `save_assets()`.",
    (Tokenizer, "merges"): "Saved as an asset file by `save_assets()`.",
    (Task, "compile"): "Only says whether to compile with the defaults.",
}

# Constructor arguments that `get_config()` leaves out today, as
# `{module: {class name: [arguments]}}`, from a scan on 2026-11-13. This list
# may only shrink: serialize the argument and remove its entry here.
KNOWN_GAPS = {
    "keras_hub.src.models.blip2.blip2_seq_2_seq_lm_preprocessor": {
        "BLIP2Seq2SeqLMPreprocessor": [
            "encoder_sequence_length",
            "decoder_sequence_length",
        ],
    },
    "keras_hub.src.models.clip.clip_layers": {
        "CLIPVisionEmbedding": ["data_format"],
    },
    "keras_hub.src.models.clip.clip_vision_encoder": {
        "CLIPVisionEncoder": ["data_format"],
    },
    "keras_hub.src.models.d_fine.d_fine_object_detector": {
        "DFineObjectDetector": ["activation"],
    },
    "keras_hub.src.models.deit.deit_image_classifier": {
        "DeiTImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.deit.deit_layers": {
        "DeiTEmbeddings": ["data_format"],
    },
    "keras_hub.src.models.depth_anything.depth_anything_backbone": {
        "DepthAnythingBackbone": ["data_format"],
    },
    "keras_hub.src.models.depth_anything.depth_anything_layers": {
        "DepthAnythingDepthEstimationHead": ["data_format"],
        "DepthAnythingFeatureFusionLayer": ["data_format"],
        "DepthAnythingNeck": ["data_format"],
        "DepthAnythingPreActResidualLayer": ["data_format"],
        "DepthAnythingReassembleLayer": ["data_format"],
        "DepthAnythingTokenToImage": ["data_format"],
    },
    "keras_hub.src.models.dinov2.dinov2_backbone": {
        "DINOV2Backbone": ["data_format"],
    },
    "keras_hub.src.models.dinov2.dinov2_layers": {
        "DINOV2Embedding": ["data_format"],
        "DINOV2PatchEmbedding": ["data_format"],
    },
    "keras_hub.src.models.dinov3.dinov3_layers": {
        "DINOV3Attention": [
            "use_query_bias",
            "use_key_bias",
            "use_value_bias",
            "use_proj_bias",
        ],
        "DINOV3Embedding": ["data_format"],
        "DINOV3PatchEmbedding": ["data_format"],
        "DINOV3RopePositionEmbedding": ["data_format"],
    },
    "keras_hub.src.models.efficientnet.efficientnet_backbone": {
        "EfficientNetBackbone": ["stackwise_nores_option"],
    },
    "keras_hub.src.models.gemma4.gemma4_attention": {
        "Gemma4TextAttention": ["kernel_initializer"],
        "Gemma4VisionAttention": ["use_clipped_linears"],
    },
    "keras_hub.src.models.gemma4.gemma4_causal_lm": {
        "Gemma4CausalLM": ["final_logit_cap"],
    },
    "keras_hub.src.models.gemma4.gemma4_decoder_block": {
        "Gemma4VisionDecoderBlock": ["use_clipped_linears"],
    },
    "keras_hub.src.models.gemma4.gemma4_image_converter": {
        "Gemma4ImageConverter": ["max_soft_tokens", "pooling_kernel_size"],
    },
    "keras_hub.src.models.gemma4.gemma4_vision_encoder": {
        "Gemma4VisionAveragePooling": ["image_size"],
    },
    "keras_hub.src.models.hgnetv2.hgnetv2_image_classifier": {
        "HGNetV2ImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.image_classifier": {
        "ImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.image_segmenter_preprocessor": {
        "ImageSegmenterPreprocessor": ["resize_output_mask"],
    },
    "keras_hub.src.models.llama.llama_attention": {
        "LlamaAttention": [
            "rope_position_scaling_factor",
            "rope_frequency_adjustment_factor",
        ],
    },
    "keras_hub.src.models.llama.llama_backbone": {
        "LlamaBackbone": ["tie_word_embeddings"],
    },
    "keras_hub.src.models.llama.llama_decoder": {
        "LlamaTransformerDecoder": [
            "rope_position_scaling_factor",
            "rope_frequency_adjustment_factor",
        ],
    },
    "keras_hub.src.models.llama.llama_rotary_embedding": {
        "LlamaRotaryEmbedding": ["pretraining_sequence_length"],
    },
    "keras_hub.src.models.llama3.llama3_tokenizer": {
        "Llama3Tokenizer": ["bos_token", "eos_token", "misc_special_tokens"],
    },
    "keras_hub.src.models.metaclip_2.metaclip_2_layers": {
        "MetaCLIP2VisionEmbedding": ["data_format"],
    },
    "keras_hub.src.models.metaclip_2.metaclip_2_vision_encoder": {
        "MetaCLIP2VisionEncoder": ["data_format"],
    },
    "keras_hub.src.models.mit.mit_layers": {
        "HierarchicalTransformerEncoder": [
            "sr_ratio",
            "drop_prob",
            "layer_norm_epsilon",
        ],
    },
    "keras_hub.src.models.mixtral.mixtral_backbone": {
        "MixtralBackbone": ["output_router_logits"],
    },
    "keras_hub.src.models.mixtral.mixtral_decoder": {
        "MixtralTransformerDecoder": [
            "output_router_logits",
            "router_aux_loss_coef",
        ],
    },
    "keras_hub.src.models.mobilenet.mobilenet_image_classifier": {
        "MobileNetImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.mobilenetv5.mobilenetv5_image_classifier": {
        "MobileNetV5ImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.mobilenetv5.mobilenetv5_utils": {
        "SelectAdaptivePool2d": ["channel_axis"],
    },
    "keras_hub.src.models.moonshine.moonshine_audio_to_text_preprocessor": {
        "MoonshineAudioToTextPreprocessor": ["decoder_sequence_length"],
    },
    "keras_hub.src.models.parseq.parseq_tokenizer": {
        "PARSeqTokenizer": [
            "remove_whitespace",
            "normalize_unicode",
            "max_label_length",
        ],
    },
    "keras_hub.src.models.qwen.qwen_decoder": {
        "QwenTransformerDecoder": [
            "use_sliding_window_attention",
            "sliding_window_size",
        ],
    },
    "keras_hub.src.models.qwen3.qwen3_attention": {
        "Qwen3Attention": ["head_dim", "layer_norm_epsilon"],
    },
    "keras_hub.src.models.qwen3.qwen3_decoder": {
        "Qwen3TransformerDecoder": ["head_dim", "sliding_window_size"],
    },
    "keras_hub.src.models.qwen3.qwen3_layernorm": {
        "Qwen3LayerNorm": ["head_dim"],
    },
    "keras_hub.src.models.qwen_moe.qwen_moe_backbone": {
        "QwenMoeBackbone": ["router_aux_loss_coefficient"],
    },
    "keras_hub.src.models.qwen_moe.qwen_moe_decoder": {
        "QwenMoeTransformerDecoder": [
            "activation",
            "kernel_initializer",
            "layer_index",
        ],
    },
    "keras_hub.src.models.retinanet.prediction_head": {
        "PredictionHead": ["data_format"],
    },
    "keras_hub.src.models.retinanet.retinanet_backbone": {
        "RetinaNetBackbone": ["data_format"],
    },
    "keras_hub.src.models.retinanet.retinanet_label_encoder": {
        "RetinaNetLabelEncoder": [
            "box_matcher_match_values",
            "box_matcher_force_match_for_each_col",
        ],
    },
    "keras_hub.src.models.retinanet.retinanet_object_detector": {
        "RetinaNetObjectDetector": [
            "classification_head_prior_probability",
            "activation",
        ],
    },
    "keras_hub.src.models.roformer_v2.roformer_v2_attention": {
        "RoformrPositionalEmbedding": ["output_dim"],
    },
    "keras_hub.src.models.sam3.sam3_layers": {
        "SAM3Embedding": ["data_format"],
        "SAM3PatchEmbedding": ["data_format"],
    },
    "keras_hub.src.models.sam3.sam3_mask_decoder": {
        "SAM3MaskDecoder": ["data_format"],
        "SAM3PixelDecoder": ["data_format"],
    },
    "keras_hub.src.models.sam3.sam3_pc_image_segmenter_preprocessor": {
        "SAM3PromptableConceptImageSegmenterPreprocessor": ["point_pad_value"],
    },
    "keras_hub.src.models.seq_2_seq_lm_preprocessor": {
        "Seq2SeqLMPreprocessor": [
            "encoder_sequence_length",
            "decoder_sequence_length",
        ],
    },
    "keras_hub.src.models.siglip.siglip_layers": {
        "SigLIPVisionEmbedding": ["data_format"],
    },
    "keras_hub.src.models.siglip.siglip_vision_encoder": {
        "SigLIPVisionEncoder": ["data_format"],
    },
    "keras_hub.src.models.stable_diffusion_3.mmdit": {
        "MMDiT": ["data_format"],
        "PatchEmbedding": ["data_format"],
    },
    "keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone": {
        "StableDiffusion3Backbone": ["data_format"],
    },
    "keras_hub.src.models.swin_transformer.swin_transformer_image_classifier": {
        "SwinTransformerImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.swin_transformer.swin_transformer_layers": {
        "SwinTransformerBlock": [
            "qkv_bias",
            "dropout_rate",
            "attention_dropout",
            "drop_path",
            "act_layer",
            "norm_layer",
        ],
        "WindowAttention": ["qk_scale"],
    },
    "keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm_preprocessor": {
        "T5GemmaSeq2SeqLMPreprocessor": [
            "encoder_sequence_length",
            "decoder_sequence_length",
        ],
    },
    "keras_hub.src.models.t5gemma2.t5gemma2_attention": {
        "T5Gemma2Attention": ["rope_scaling_factor"],
        "T5Gemma2MergedAttention": ["rope_scaling_factor"],
    },
    "keras_hub.src.models.t5gemma2.t5gemma2_decoder": {
        "T5Gemma2DecoderLayer": ["rope_scaling_factor"],
    },
    "keras_hub.src.models.t5gemma2.t5gemma2_encoder": {
        "T5Gemma2EncoderLayer": ["rope_scaling_factor"],
    },
    "keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm_preprocessor": {
        "T5Gemma2Seq2SeqLMPreprocessor": [
            "encoder_sequence_length",
            "decoder_sequence_length",
        ],
    },
    "keras_hub.src.models.vae.vae_layers": {
        "Conv2DMultiHeadAttention": ["data_format"],
        "DiagonalGaussianDistributionSampler": ["method"],
        "ResNetBlock": ["data_format"],
        "VAEDecoder": ["data_format"],
        "VAEEncoder": ["data_format"],
    },
    "keras_hub.src.models.vgg.vgg_image_classifier": {
        "VGGImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.video_prism.video_prism_layers": {
        "VideoPrismFactorizedReshape": ["data_format"],
    },
    "keras_hub.src.models.vit.vit_image_classifier": {
        "ViTImageClassifier": ["head_dtype"],
    },
    "keras_hub.src.models.vit.vit_layers": {
        "ViTPatchingAndEmbedding": ["use_patch_bias", "data_format"],
    },
    "keras_hub.src.tokenizers.word_piece_tokenizer": {
        "WordPieceTokenizer": ["split_on_cjk"],
    },
}


def literal_keys(function):
    """Returns every key that `function` writes as a literal."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys.update(
                k.value for k in node.keys if isinstance(k, ast.Constant)
            )
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant):
                keys.add(node.slice.value)
        elif isinstance(node, ast.Call):
            keys.update(k.arg for k in node.keywords if k.arg)
    return keys


def is_keras_hub(cls):
    return cls.__module__.startswith("keras_hub.")


def is_intentional(cls, argument):
    return any(
        issubclass(cls, base) and argument == name for base, name in INTENTIONAL
    )


def get_config_gaps():
    """Scans `keras_hub.src` (tests excluded) for unserialized arguments.

    A class is scanned if it defines `__init__` and its nearest `get_config()`
    is keras-hub code. An argument is a gap if no keras-hub `get_config()` in
    the MRO writes it as a literal key.

    Returns:
        A tuple `(gaps, unimported)`: `{class path: [arguments]}`, and the
        modules that could not be imported in this environment.
    """
    gaps = {}
    unimported = set()
    modules = pkgutil.walk_packages([str(SRC_ROOT)], "keras_hub.src.")
    for module_name in sorted(module.name for module in modules):
        if module_name.endswith("_test") or ".tests." in module_name:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            unimported.add(module_name)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name or "__init__" not in cls.__dict__:
                continue
            owners = [c for c in cls.__mro__ if "get_config" in c.__dict__]
            if not owners or not is_keras_hub(owners[0]):
                continue
            keys = set()
            for owner in filter(is_keras_hub, owners):
                keys |= literal_keys(owner.__dict__["get_config"])
            parameters = inspect.signature(cls.__init__).parameters.values()
            missing = [
                p.name
                for p in parameters
                if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                and p.name not in SKIPPED_ARGUMENTS
                and p.name not in keys
                and not is_intentional(cls, p.name)
            ]
            if missing:
                gaps[f"{module_name}.{cls.__name__}"] = missing
    return gaps, unimported


class GetConfigCoverageTest(TestCase):
    @pytest.mark.skipif(
        keras.config.backend() == "openvino",
        reason="The scan does not depend on the backend.",
    )
    def test_get_config_returns_every_constructor_argument(self):
        gaps, unimported = get_config_gaps()
        known_gaps = {
            f"{module}.{name}": arguments
            for module, classes in KNOWN_GAPS.items()
            for name, arguments in classes.items()
        }
        new = []
        fixed = []
        for path in sorted(set(gaps) | set(known_gaps)):
            # Gaps in modules that cannot be imported here are not judged.
            if any(path.startswith(f"{module}.") for module in unimported):
                continue
            found = gaps.get(path, [])
            known = known_gaps.get(path, [])
            new += [f"  {path}: {arg}" for arg in found if arg not in known]
            fixed += [f"  {path}: {arg}" for arg in known if arg not in found]
        message = []
        if new:
            message += [
                "No keras-hub `get_config()` of these classes or their "
                "parents has these constructor arguments as a literal key, "
                "so `from_config()` silently rebuilds the object with their "
                "default values. Store each one in `__init__` and add it to "
                "`get_config()`, as "
                "https://github.com/keras-team/keras-hub/pull/3200 does for "
                "`data_format`:"
            ] + new
        if fixed:
            message += [
                "These arguments are serialized now; remove them from "
                "KNOWN_GAPS:"
            ] + fixed
        if message:
            self.fail("\n".join(message))
