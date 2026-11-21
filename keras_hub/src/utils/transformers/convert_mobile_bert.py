import numpy as np

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    NORMALIZATION_TYPES,
)
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.models.text_embedder_preprocessor import (
    TextEmbedderPreprocessor,
)
from keras_hub.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import SAFETENSOR_FILE
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_modern_bert import (
    _read_pooling_mode,
)
from keras_hub.src.utils.transformers.safetensor_utils import safetensors

backbone_cls = MobileBertBackbone

# HF task classes built with `MobileBertModel(config, add_pooling_layer=False)`
# ship no `pooler.dense.*` even when `classifier_activation=True`.
_HF_ARCHITECTURES_WITHOUT_POOLER = (
    "MobileBertForMaskedLM",
    "MobileBertForQuestionAnswering",
    "MobileBertForTokenClassification",
)
# HF task classes whose checkpoints carry the `cls.predictions.*` MLM head.
_HF_ARCHITECTURES_WITH_MLM_HEAD = (
    "MobileBertForPreTraining",
    "MobileBertForMaskedLM",
)


def _hf_architecture(transformers_config):
    if (
        "architectures" not in transformers_config
        or not transformers_config["architectures"]
    ):
        return None
    return transformers_config["architectures"][0]


def _hf_has_pooler(transformers_config):
    return (
        transformers_config["classifier_activation"]
        and _hf_architecture(transformers_config)
        not in _HF_ARCHITECTURES_WITHOUT_POOLER
    )


def _has_hf_weight(loader, hf_weight_key):
    if hasattr(loader, "state_dict"):
        keys = loader.state_dict.keys()
    elif loader.safetensor_config is not None:
        keys = loader.safetensor_config["weight_map"].keys()
    else:
        fname = loader.fname if loader.fname is not None else SAFETENSOR_FILE
        if fname not in loader.safetensor_files:
            if safetensors is None or not check_file_exists(
                loader.preset, fname
            ):
                return False
            path = get_file(loader.preset, fname)
            loader.safetensor_files[fname] = loader.enter_context(
                safetensors.safe_open(path, framework="np")
            )
        keys = loader.safetensor_files[fname].keys()
    return hf_weight_key in keys or f"mobilebert.{hf_weight_key}" in keys


def convert_backbone_config(transformers_config):
    if not transformers_config["trigram_input"]:
        raise ValueError("MobileBERT requires trigram_input=True")
    if not transformers_config["use_bottleneck"]:
        raise ValueError("MobileBERT requires use_bottleneck=True")
    if not transformers_config["key_query_shared_bottleneck"]:
        raise ValueError("MobileBERT requires key_query_shared_bottleneck=True")
    if transformers_config["use_bottleneck_attention"]:
        raise ValueError("MobileBERT requires use_bottleneck_attention=False")
    if (
        transformers_config["true_hidden_size"]
        != transformers_config["intra_bottleneck_size"]
    ):
        raise ValueError(
            "MobileBERT requires true_hidden_size == intra_bottleneck_size"
        )
    if transformers_config["normalization_type"] not in NORMALIZATION_TYPES:
        raise ValueError(
            f"MobileBERT requires normalization_type in {NORMALIZATION_TYPES}, "
            f"got {transformers_config['normalization_type']!r}"
        )
    if transformers_config["num_feedforward_networks"] < 1:
        raise ValueError(
            "`num_feedforward_networks` must be >= 1. Received: "
            "num_feedforward_networks="
            f"{transformers_config['num_feedforward_networks']}."
        )

    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "embedding_dim": transformers_config["embedding_size"],
        "intra_bottleneck_dim": transformers_config["intra_bottleneck_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_feedforward_networks": transformers_config[
            "num_feedforward_networks"
        ],
        "activation": transformers_config["hidden_act"],
        "normalization_type": transformers_config["normalization_type"],
        # HF uses 1e-5 for emb and output norm by default. KerasHub uses
        # layer_norm_epsilon uniformly.
        "layer_norm_epsilon": transformers_config["layer_norm_eps"],
        "use_pooler_dense": _hf_has_pooler(transformers_config),
        "dropout": transformers_config["hidden_dropout_prob"],
        "attention_dropout": transformers_config[
            "attention_probs_dropout_prob"
        ],
        "max_sequence_length": transformers_config["max_position_embeddings"],
        "num_segments": transformers_config["type_vocab_size"],
    }


def transpose_and_reshape(x, shape):
    return np.reshape(np.transpose(x), shape)


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embedding_projection").dense.kernel,
        hf_weight_key="embeddings.embedding_transformation.weight",
        hook_fn=transpose_and_reshape,
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embedding_projection").dense.bias,
        hf_weight_key="embeddings.embedding_transformation.bias",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer(
            "position_embedding"
        ).position_embeddings,
        hf_weight_key="embeddings.position_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("segment_embedding").embeddings,
        hf_weight_key="embeddings.token_type_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embeddings_layer_norm").gamma,
        hf_weight_key="embeddings.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embeddings_layer_norm").beta,
        hf_weight_key="embeddings.LayerNorm.bias",
    )

    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"transformer_layer_{i}")
        hf_prefix = f"encoder.layer.{i}."

        # Bottleneck input
        loader.port_weight(
            keras_variable=block.bottleneck_input.dense.kernel,
            hf_weight_key=f"{hf_prefix}bottleneck.input.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block.bottleneck_input.dense.bias,
            hf_weight_key=f"{hf_prefix}bottleneck.input.dense.bias",
        )
        loader.port_weight(
            keras_variable=block.bottleneck_input.layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}bottleneck.input.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=block.bottleneck_input.layer_norm.beta,
            hf_weight_key=f"{hf_prefix}bottleneck.input.LayerNorm.bias",
        )

        # Bottleneck attention
        loader.port_weight(
            keras_variable=block.bottleneck_attention.dense.kernel,
            hf_weight_key=f"{hf_prefix}bottleneck.attention.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block.bottleneck_attention.dense.bias,
            hf_weight_key=f"{hf_prefix}bottleneck.attention.dense.bias",
        )
        loader.port_weight(
            keras_variable=block.bottleneck_attention.layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}bottleneck.attention.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=block.bottleneck_attention.layer_norm.beta,
            hf_weight_key=f"{hf_prefix}bottleneck.attention.LayerNorm.bias",
        )

        # Attention
        attn = block.attention
        loader.port_weight(
            keras_variable=attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.self.query.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.query_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.self.query.bias",
        )
        loader.port_weight(
            keras_variable=attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.self.key.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.key_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.self.key.bias",
        )
        loader.port_weight(
            keras_variable=attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.self.value.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.value_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.self.value.bias",
        )
        loader.port_weight(
            keras_variable=attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.output_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.output.dense.bias",
        )
        loader.port_weight(
            keras_variable=attn.output_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}attention.output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=attn.output_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}attention.output.LayerNorm.bias",
        )

        # FFNs (j=0..2 are ffn.j)
        num_ffn = backbone.num_feedforward_networks
        for j in range(num_ffn - 1):
            ffn = block.ffn_layers[j]
            loader.port_weight(
                keras_variable=ffn.intermediate_dense.kernel,
                hf_weight_key=f"{hf_prefix}ffn.{j}.intermediate.dense.weight",
                hook_fn=transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=ffn.intermediate_dense.bias,
                hf_weight_key=f"{hf_prefix}ffn.{j}.intermediate.dense.bias",
            )
            loader.port_weight(
                keras_variable=ffn.output_dense.kernel,
                hf_weight_key=f"{hf_prefix}ffn.{j}.output.dense.weight",
                hook_fn=transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=ffn.output_dense.bias,
                hf_weight_key=f"{hf_prefix}ffn.{j}.output.dense.bias",
            )
            loader.port_weight(
                keras_variable=ffn.output_layer_norm.gamma,
                hf_weight_key=f"{hf_prefix}ffn.{j}.output.LayerNorm.weight",
            )
            loader.port_weight(
                keras_variable=ffn.output_layer_norm.beta,
                hf_weight_key=f"{hf_prefix}ffn.{j}.output.LayerNorm.bias",
            )

        # FFN_3 (intermediate + output)
        ffn_last = block.ffn_layers[-1]
        loader.port_weight(
            keras_variable=ffn_last.intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}intermediate.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=ffn_last.intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=ffn_last.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=ffn_last.output_dense.bias,
            hf_weight_key=f"{hf_prefix}output.dense.bias",
        )
        loader.port_weight(
            keras_variable=ffn_last.output_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=ffn_last.output_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}output.LayerNorm.bias",
        )

        # Output Bottleneck
        loader.port_weight(
            keras_variable=block.output_bottleneck_dense.kernel,
            hf_weight_key=f"{hf_prefix}output.bottleneck.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block.output_bottleneck_dense.bias,
            hf_weight_key=f"{hf_prefix}output.bottleneck.dense.bias",
        )
        loader.port_weight(
            keras_variable=block.output_bottleneck_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}output.bottleneck.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=block.output_bottleneck_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}output.bottleneck.LayerNorm.bias",
        )

    if (
        backbone.pooled_dense is not None
        and _hf_has_pooler(transformers_config)
        and _has_hf_weight(loader, "pooler.dense.weight")
    ):
        loader.port_weight(
            keras_variable=backbone.get_layer("pooled_dense").kernel,
            hf_weight_key="pooler.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=backbone.get_layer("pooled_dense").bias,
            hf_weight_key="pooler.dense.bias",
        )


def convert_head(task, loader, transformers_config):
    arch = _hf_architecture(transformers_config)
    if arch is not None and arch not in _HF_ARCHITECTURES_WITH_MLM_HEAD:
        # e.g. `MobileBertForSequenceClassification`: the backbone is ported,
        # the MLM head keeps its fresh initialization.
        return
    if arch is None and not _has_hf_weight(loader, "cls.predictions.bias"):
        return

    loader.port_weight(
        keras_variable=task.masked_lm_head.transform_dense.kernel,
        hf_weight_key="cls.predictions.transform.dense.weight",
        hook_fn=transpose_and_reshape,
    )
    loader.port_weight(
        keras_variable=task.masked_lm_head.transform_dense.bias,
        hf_weight_key="cls.predictions.transform.dense.bias",
    )
    loader.port_weight(
        keras_variable=task.masked_lm_head.transform_layer_norm.gamma,
        hf_weight_key="cls.predictions.transform.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=task.masked_lm_head.transform_layer_norm.beta,
        hf_weight_key="cls.predictions.transform.LayerNorm.bias",
    )
    # The cls.predictions.dense.weight is (384, V), which aligns with the
    # un-transposed kernel needed when concat happens along the last dim.
    # We do NOT transpose it.
    loader.port_weight(
        keras_variable=task.masked_lm_head.extra_kernel,
        hf_weight_key="cls.predictions.dense.weight",
    )
    loader.port_weight(
        keras_variable=task.masked_lm_head.bias,
        hf_weight_key="cls.predictions.bias",
    )


# (keras_hub kwarg, tokenizer_config.json key, tokenizer.json section, key).
# HF writes the slow-tokenizer config to `tokenizer_config.json` and the
# fast-tokenizer config to `tokenizer.json`; a key missing (or `null`) in
# the first falls through to the second. When neither file sets a key we
# follow `AutoTokenizer` for `model_type="mobilebert"` (`do_lower_case=True`,
# `strip_accents=None` -> follows lowercase), not
# `PreTrainedTokenizerFast(tokenizer_file=...)`, which treats a missing
# normalizer as cased / accent-preserving.
_HF_TOKENIZER_FIELDS = (
    ("lowercase", "do_lower_case", "normalizer", "lowercase"),
    ("strip_accents", "strip_accents", "normalizer", "strip_accents"),
    (
        "split_on_cjk",
        "tokenize_chinese_chars",
        "normalizer",
        "handle_chinese_chars",
    ),
    ("oov_token", "unk_token", "model", "unk_token"),
    (
        "suffix_indicator",
        "wordpieces_prefix",
        "model",
        "continuing_subword_prefix",
    ),
)


def _token_string(value):
    # Special tokens may be serialized as `AddedToken` dicts.
    return value["content"] if isinstance(value, dict) else value


def convert_tokenizer(cls, preset, **kwargs):
    slow_config = {}
    if check_file_exists(preset, HF_TOKENIZER_CONFIG_FILE):
        slow_config = load_json(preset, HF_TOKENIZER_CONFIG_FILE) or {}
    fast_config = {}
    if check_file_exists(preset, "tokenizer.json"):
        fast_config = load_json(preset, "tokenizer.json") or {}
    config = {}
    for kh_key, slow_key, section, fast_key in _HF_TOKENIZER_FIELDS:
        value = slow_config[slow_key] if slow_key in slow_config else None
        if (
            value is None
            and section in fast_config
            and isinstance(fast_config[section], dict)
            and fast_key in fast_config[section]
        ):
            value = fast_config[section][fast_key]
        if value is not None:
            config[kh_key] = _token_string(value)
    config.setdefault("lowercase", True)
    # Caller kwargs win over anything read from the checkpoint, and an
    # explicit `lowercase` override also drives default `strip_accents`.
    config.update(kwargs)
    config.setdefault("strip_accents", config["lowercase"])
    if "vocabulary" not in config:
        if check_file_exists(preset, "vocab.txt"):
            config["vocabulary"] = get_file(preset, "vocab.txt")
        elif (
            "model" in fast_config
            and isinstance(fast_config["model"], dict)
            and "vocab" in fast_config["model"]
            and isinstance(fast_config["model"]["vocab"], dict)
        ):
            vocab = dict(fast_config["model"]["vocab"])
            if "added_tokens" in fast_config and isinstance(
                fast_config["added_tokens"], list
            ):
                for tok_obj in fast_config["added_tokens"]:
                    if (
                        isinstance(tok_obj, dict)
                        and "content" in tok_obj
                        and "id" in tok_obj
                    ):
                        vocab[tok_obj["content"]] = tok_obj["id"]
            config["vocabulary"] = [
                tok
                for tok, _ in sorted(vocab.items(), key=lambda item: item[1])
            ]
        else:
            config["vocabulary"] = get_file(preset, "vocab.txt")
    return cls(**config)


def load_task_config(preset, transformers_config, cls=None):
    kwargs = {}
    if cls is None or issubclass(cls, TextEmbedder):
        if check_file_exists(preset, "modules.json"):
            modules = load_json(preset, "modules.json") or []
            kwargs["normalize"] = any(
                isinstance(m, dict)
                and "type" in m
                and isinstance(m["type"], str)
                and "Normalize" in m["type"]
                for m in modules
            )
        if check_file_exists(preset, "1_Pooling/config.json"):
            pooling_config = load_json(preset, "1_Pooling/config.json") or {}
            kwargs["pooling_mode"] = _read_pooling_mode(pooling_config)
    if cls is not None and issubclass(cls, TextClassifier):
        dropout = (
            transformers_config["classifier_dropout"]
            if "classifier_dropout" in transformers_config
            else None
        )
        if dropout is None and "hidden_dropout_prob" in transformers_config:
            dropout = transformers_config["hidden_dropout_prob"]
        if dropout is not None:
            kwargs["dropout"] = dropout
    return kwargs


def load_preprocessor_config(preset, transformers_config, cls=None):
    max_pos = (
        transformers_config["max_position_embeddings"]
        if "max_position_embeddings" in transformers_config
        else None
    )
    if check_file_exists(preset, "sentence_bert_config.json"):
        st_config = load_json(preset, "sentence_bert_config.json") or {}
        if (
            "max_seq_length" in st_config
            and st_config["max_seq_length"] is not None
        ):
            seq_len = st_config["max_seq_length"]
            if max_pos is not None:
                seq_len = min(seq_len, max_pos)
            return {"sequence_length": seq_len}

    if check_file_exists(preset, "modules.json") and max_pos is not None:
        seq_len = max_pos
        if check_file_exists(preset, HF_TOKENIZER_CONFIG_FILE):
            tok_cfg = load_json(preset, HF_TOKENIZER_CONFIG_FILE) or {}
            if (
                "model_max_length" in tok_cfg
                and tok_cfg["model_max_length"] is not None
            ):
                seq_len = min(seq_len, int(tok_cfg["model_max_length"]))
        return {"sequence_length": seq_len}

    default_seq_len = (
        256
        if cls is not None and issubclass(cls, TextEmbedderPreprocessor)
        else 512
    )
    if max_pos is not None and max_pos < default_seq_len:
        return {"sequence_length": max_pos}
    return {}
