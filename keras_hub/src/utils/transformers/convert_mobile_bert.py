import numpy as np

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = MobileBertBackbone


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
        "use_pooler_dense": transformers_config["classifier_activation"],
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

    if transformers_config["classifier_activation"]:
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
    del transformers_config

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


def convert_tokenizer(cls, preset, **kwargs):
    transformers_config = load_json(preset, "tokenizer_config.json")
    return cls(
        get_file(preset, "vocab.txt"),
        lowercase=transformers_config["do_lower_case"],
        **kwargs,
    )
