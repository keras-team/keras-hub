import numpy as np

from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = BertBackbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
    }


def convert_weights(backbone, loader, transformers_config):
    # Detect LayerNorm naming convention. Standard BERT uses
    # "gamma/beta", sentence-transformers use "weight/bias".
    # `loader.get_tensor()` handles `bert.` prefix auto-detection
    # via `SafetensorLoader.get_prefixed_key()` suffix matching.
    try:
        loader.get_tensor("embeddings.LayerNorm.gamma")
        ln_gamma, ln_beta = "gamma", "beta"
    except Exception:
        ln_gamma, ln_beta = "weight", "bias"

    # Embedding layer
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="embeddings.word_embeddings.weight",
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
        keras_variable=backbone.get_layer("embeddings_layer_norm").beta,
        hf_weight_key=f"embeddings.LayerNorm.{ln_beta}",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embeddings_layer_norm").gamma,
        hf_weight_key=f"embeddings.LayerNorm.{ln_gamma}",
    )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    # Attention blocks
    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"transformer_layer_{i}")
        attn = block._self_attention_layer
        hf_prefix = "encoder.layer."
        # Attention layers
        loader.port_weight(
            keras_variable=attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.query.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.query_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.query.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.key.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.key_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.key.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.value.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.value_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.value.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.output_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        # Attention layer norm.
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.LayerNorm.{ln_beta}",
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.LayerNorm.{ln_gamma}",
        )
        # MLP layers
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=block._feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block._feedforward_output_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.bias",
        )
        # Output layer norm.
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.output.LayerNorm.{ln_beta}",
        )
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.output.LayerNorm.{ln_gamma}",
        )

    loader.port_weight(
        keras_variable=backbone.get_layer("pooled_dense").kernel,
        hf_weight_key="pooler.dense.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("pooled_dense").bias,
        hf_weight_key="pooler.dense.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    transformers_config = load_json(preset, HF_TOKENIZER_CONFIG_FILE)
    # Some BERT-architecture models (e.g. multilingual-e5) ship an XLM-RoBERTa
    # SentencePiece tokenizer instead of a WordPiece vocab.txt.
    if "XLMRoberta" in transformers_config.get("tokenizer_class", ""):
        return XLMRobertaTokenizer(
            proto=get_file(preset, "sentencepiece.bpe.model"), **kwargs
        )
    return cls(
        get_file(preset, "vocab.txt"),
        lowercase=transformers_config["do_lower_case"],
        **kwargs,
    )


def load_task_config(preset, transformers_config):
    """Detect sentence-transformer pipeline settings.

    Reads two sentence-transformer config files when available:

    - ``modules.json``: determines whether L2 normalization is applied.
      Models like all-MiniLM-* include a Normalize module, while
      paraphrase-* models do not.
    - ``1_Pooling/config.json``: determines the pooling strategy
      (mean, cls, or max).

    Returns kwargs suitable for ``BertTextEmbedder``.
    """
    kwargs = {}

    # Detect normalization from modules.json.
    if check_file_exists(preset, "modules.json"):
        modules = load_json(preset, "modules.json")
        kwargs["normalize"] = any(
            "Normalize" in m.get("type", "") for m in modules
        )

    # Detect pooling mode from 1_Pooling/config.json.
    pooling_config_file = "1_Pooling/config.json"
    if check_file_exists(preset, pooling_config_file):
        pooling_config = load_json(preset, pooling_config_file)
        if pooling_config.get("pooling_mode_cls_token", False):
            kwargs["pooling_mode"] = "cls"
        elif pooling_config.get("pooling_mode_max_tokens", False):
            kwargs["pooling_mode"] = "max"
        else:
            kwargs["pooling_mode"] = "mean"

    return kwargs


def load_preprocessor_config(preset, transformers_config):
    """Read sentence-transformer sequence length from sentence_bert_config.json.

    Sentence-transformer models define their actual max sequence length in
    ``sentence_bert_config.json`` as ``max_seq_length``, which is often
    shorter than ``max_position_embeddings`` in ``config.json``
    (e.g., 256 vs 512 for all-MiniLM-L6-v2).
    """
    st_config_file = "sentence_bert_config.json"
    if check_file_exists(preset, st_config_file):
        st_config = load_json(preset, st_config_file)
        max_seq_length = st_config.get("max_seq_length")
        if max_seq_length is not None:
            return {"sequence_length": max_seq_length}
    return {}
