import os
import tensorflow as tf
import transformers
from absl import app
from absl import flags
from tensorflow import keras
import tensorflow_models as tfm
import keras_nlp
import json

FLAGS = flags.FLAGS

PRESET_MAP = {
    "bert_tiny_uncased_en": ("roberta.base", "bert_tiny_uncased_en")
}

flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)
VOCAB_SIZE = 30522
NUM_LAYERS = 24
NUM_ATTN_HEADS = 16
EMBEDDING_SIZE = 1024

MODEL_SUFFIX = "uncased"
# MODEL_SPEC_STR = "L-24_H-1024_A-16"
MODEL_SPEC_STR = "L-2_H-128_A-2"
model_source = {
    'bert_tiny_en_uncased': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_tiny_uncased_en.ipynb',
    'bert_small_en_uncased': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_small_uncased_en.ipynb',
    'bert_medium_en_uncased': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_medium_uncased_en.ipynb',
    'bert_base_en_uncased': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_base_uncased.ipynb',
    'bert_base_en': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_base_cased.ipynb',
    'bert_base_zh': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_base_zh.ipynb',
    'bert_base_multi': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_base_multi_cased.ipynb',
    'bert_large_en_uncased': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_large_uncased_en.ipynb',
    'bert_large_en': 'https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/bert_large_cased_en.ipynb'}


def download_model(preset, TOKEN_TYPE, hf_model_name):
    print("-> Download original weights in " + preset)
    zip_path = f"""https://storage.googleapis.com/tf_model_garden/nlp/bert/v3/{MODEL_SUFFIX}_L-12_H-768_A-12.tar.gz"""
    path_file = keras.utils.get_file(
        f"""/home/x000ff4/open_source/486/keras-nlp/tools/checkpoint_conversion/{preset}.tar.gz""",
        zip_path,
        extract=False,
        archive_format="tar",
    )
    print('path_file', path_file)
    extract_dir = f"./content/{preset}/tmp/temp_dir/raw/"
    vocab_path = os.path.join(extract_dir, "vocab.txt")
    checkpoint_path = os.path.join(extract_dir, "bert_model.ckpt")
    config_path = os.path.join(extract_dir, "bert_config.json")

    os.system(f"tar -C ./content/{preset} -xvf {path_file}")

    vars = tf.train.list_variables(checkpoint_path)
    weights = {}
    for name, shape in vars:
        weight = tf.train.load_variable(checkpoint_path, name)
        weights[name] = weight
        print(name)
    model = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased",
                                                      load_weights=True)  # keras_nlp.models.BertBase(vocabulary_size=VOCAB_SIZE)
    model.summary()
    return vocab_path, checkpoint_path, config_path, weights, model


def convert_checkpoints(preset, weights, model):
    print("\n-> Convert original weights to KerasNLP format.")

    # Transformer layers.
    if preset in ['bert_base_en_uncased', 'bert_base_en']:
        model.get_layer("token_embedding").embeddings.assign(
            weights[
                "encoder/layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
            ]
        )
        model.get_layer("position_embedding").position_embeddings.assign(
            weights[
                "encoder/layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
            ]
        )
        model.get_layer("segment_embedding").embeddings.assign(
            weights[
                "encoder/layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
            ]
        )
        model.get_layer("embeddings_layer_norm").gamma.assign(
            weights["encoder/layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE"]
        )
        model.get_layer("embeddings_layer_norm").beta.assign(
            weights["encoder/layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE"]
        )

        for i in range(model.num_layers):
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._key_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_key_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._key_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_key_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_query_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_query_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_value_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_value_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            if preset == 'bert_base_en_uncased':
                model.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._output_dense.kernel.assign(
                    weights[
                        f"encoder/layer_with_weights-{i + 4}/_attention_output_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                    ]
                )
                model.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._output_dense.bias.assign(
                    weights[
                        f"encoder/layer_with_weights-{i + 4}/_attention_output_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                    ]
                )
            else:
                model.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._output_dense.kernel.assign(
                    weights[
                        f"encoder/layer_with_weights-{i + 4}/_attention_layer/_output_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                    ]
                )
                model.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._output_dense.bias.assign(
                    weights[
                        f"encoder/layer_with_weights-{i + 4}/_attention_layer/_output_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                    ]
                )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layernorm.gamma.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layernorm.beta.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_intermediate_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_intermediate_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_intermediate_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_intermediate_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_output_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_output_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_layernorm.gamma.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_layernorm.beta.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
        if preset == 'bert_base_en_uncased':
            model.get_layer("pooled_dense").kernel.assign(
                weights["next_sentence..pooler_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"]
            )
            model.get_layer("pooled_dense").bias.assign(
                weights["next_sentence..pooler_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"]
            )
        else:
            model.get_layer("pooled_dense").kernel.assign(
                weights["encoder/layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE"]
            )
            model.get_layer("pooled_dense").bias.assign(
                weights["encoder/layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE"]
            )
        pass
    elif preset in ['bert_base_zh', 'bert_base_multi']:
        model.get_layer("token_embedding").embeddings.assign(
            weights[
                "encoder/layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
            ]
        )
        model.get_layer("position_embedding").position_embeddings.assign(
            weights[
                "encoder/layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
            ]
        )
        model.get_layer("segment_embedding").embeddings.assign(
            weights[
                "encoder/layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
            ]
        )
        model.get_layer("embeddings_layer_norm").gamma.assign(
            weights["encoder/layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE"]
        )
        model.get_layer("embeddings_layer_norm").beta.assign(
            weights["encoder/layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE"]
        )

        for i in range(model.num_layers):
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._key_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_key_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._key_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_key_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_query_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_query_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_value_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_value_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._output_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_output_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._output_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer/_output_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layernorm.gamma.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layernorm.beta.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_attention_layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_intermediate_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_intermediate_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_intermediate_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_intermediate_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_output_dense.kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_output_dense.bias.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_layernorm.gamma.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_layernorm.beta.assign(
                weights[
                    f"encoder/layer_with_weights-{i + 4}/_output_layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
        if preset == 'bert_base_zh':
            model.get_layer("pooled_dense").kernel.assign(
                weights["encoder/layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE"]
            )
            model.get_layer("pooled_dense").bias.assign(
                weights["encoder/layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE"]
            )
        else:
            model.get_layer("pooled_dense").kernel.assign(
                weights[
                    f"encoder/layer_with_weights-{model.num_layers + 4}/kernel/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
            model.get_layer("pooled_dense").bias.assign(
                weights[
                    f"encoder/layer_with_weights-{model.num_layers + 4}/bias/.ATTRIBUTES/VARIABLE_VALUE"
                ]
            )
        pass
    elif preset in ['bert_large_en_uncased', 'bert_large_en', 'bert_medium_en_uncased', 'bert_small_en_uncased',
                    'bert_tiny_en_uncased']:
        model.get_layer("token_embedding").embeddings.assign(
            weights["bert/embeddings/word_embeddings"]
        )
        model.get_layer("position_embedding").position_embeddings.assign(
            weights["bert/embeddings/position_embeddings"]
        )
        model.get_layer("segment_embedding").embeddings.assign(
            weights["bert/embeddings/token_type_embeddings"]
        )
        model.get_layer("embeddings_layer_norm").gamma.assign(
            weights["bert/embeddings/LayerNorm/gamma"]
        )
        model.get_layer("embeddings_layer_norm").beta.assign(
            weights["bert/embeddings/LayerNorm/beta"]
        )

        for i in range(model.num_layers):
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._key_dense.kernel.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/key/kernel"].reshape(
                    (EMBEDDING_SIZE, NUM_ATTN_HEADS, -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._key_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/key/bias"].reshape(
                    (NUM_ATTN_HEADS, -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.kernel.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/query/kernel"].reshape(
                    (EMBEDDING_SIZE, NUM_ATTN_HEADS, -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/query/bias"].reshape(
                    (NUM_ATTN_HEADS, -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.kernel.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/value/kernel"].reshape(
                    (EMBEDDING_SIZE, NUM_ATTN_HEADS, -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/value/bias"].reshape(
                    (NUM_ATTN_HEADS, -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._output_dense.kernel.assign(
                weights[
                    f"bert/encoder/layer_{i}/attention/output/dense/kernel"
                ].reshape((NUM_ATTN_HEADS, -1, EMBEDDING_SIZE))
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._output_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/attention/output/dense/bias"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layernorm.gamma.assign(
                weights[f"bert/encoder/layer_{i}/attention/output/LayerNorm/gamma"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layernorm.beta.assign(
                weights[f"bert/encoder/layer_{i}/attention/output/LayerNorm/beta"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_intermediate_dense.kernel.assign(
                weights[f"bert/encoder/layer_{i}/intermediate/dense/kernel"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_intermediate_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/intermediate/dense/bias"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_output_dense.kernel.assign(
                weights[f"bert/encoder/layer_{i}/output/dense/kernel"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_output_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/output/dense/bias"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_layernorm.gamma.assign(
                weights[f"bert/encoder/layer_{i}/output/LayerNorm/gamma"]
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_layernorm.beta.assign(
                weights[f"bert/encoder/layer_{i}/output/LayerNorm/beta"]
            )

        model.get_layer("pooled_dense").kernel.assign(
            weights["bert/pooler/dense/kernel"]
        )
        model.get_layer("pooled_dense").bias.assign(weights["bert/pooler/dense/bias"])
        pass

    # Save the model.
    print(f"\n-> Save KerasNLP model weights to `{preset}.h5`.")
    model.save_weights(f"{preset}.h5")

    return model


def define_preprocessor(vocab_path, checkpoint_path, config_path, model):
    def preprocess(x):
        tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=vocab_path, lowercase=False
        )
        packer = keras_nlp.layers.MultiSegmentPacker(
            sequence_length=model.max_sequence_length,
            start_value=tokenizer.token_to_id("[CLS]"),
            end_value=tokenizer.token_to_id("[SEP]"),
        )
        return packer(tokenizer(x))

    token_ids, segment_ids = preprocess(["The झटपट brown लोमड़ी."])
    encoder_config = tfm.nlp.encoders.EncoderConfig(
        type="bert",
        bert=json.load(tf.io.gfile.GFile(config_path)),
    )
    mg_model = tfm.nlp.encoders.build_encoder(encoder_config)
    checkpoint = tf.train.Checkpoint(encoder=mg_model)
    checkpoint.read(checkpoint_path).assert_consumed()
    return mg_model, token_ids, segment_ids


def check_output(
        preset,
        keras_nlp_model,
        mg_model,
        token_ids,
        segment_ids
):
    keras_nlp_output = keras_nlp_model(
        {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "padding_mask": token_ids != 0,
        }
    )["pooled_output"]

    mg_output = mg_model(
        {
            "input_word_ids": token_ids,
            "input_type_ids": segment_ids,
            "padding_mask": token_ids != 0,
        }
    )["pooled_output"]
    tf.reduce_mean(keras_nlp_output - mg_output)
    keras_nlp_model.save_weights(f"""{preset}.h5""")
    return keras_nlp_output


def main(_):
    assert (
            FLAGS.preset in PRESET_MAP.keys()
    ), f'Invalid preset {FLAGS.preset}. Must be one of {",".join(PRESET_MAP.keys())}'
    size = PRESET_MAP[FLAGS.preset][0]
    hf_model_name = PRESET_MAP[FLAGS.preset][1]

    vocab_path, checkpoint_path, config_path, weights, model = download_model(FLAGS.preset, size, hf_model_name)

    keras_nlp_model = convert_checkpoints(FLAGS.preset, weights, model)

if __name__ == "__main__":


    app.run(main)
