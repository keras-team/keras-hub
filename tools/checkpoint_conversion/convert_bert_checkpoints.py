import os
import tensorflow as tf
import transformers
from absl import app
from absl import flags
from tensorflow import keras
import tensorflow_models as tfm
import keras_nlp
import json
import shutil
import numpy as np

from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    get_md5_checksum,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "bert_base_cased": {'base': "roberta.base",
                        'base_model': "bert-base-cased",
                        'TOKEN_TYPE': "cased",
                        'MODEL_SUFFIX': "cased",
                        'MODEL_SPEC_STR': "2018_10_18/cased_L-12_H-768_A-12",
                        'MODEL_SPEC_STR_DIR': "cased_L-12_H-768_A-12/",
                        'MODEL_TYPE': "bert_base_en_cased",
                        'VOCAB_SIZE': 28996},
    "bert_base_multi_cased": {'base': "roberta.base",
                              'base_model': "bert-base-multilingual-cased",
                              'MODEL_TYPE': "bert_base_multi_cased",
                              'MODEL_SUFFIX': "multi_cased",
                              'MODEL_SPEC_STR': "2018_11_23/multi_cased_L-12_H-768_A-12",
                              'MODEL_SPEC_STR_DIR': "multi_cased_L-12_H-768_A-12/",
                              'VOCAB_SIZE': 119547},

    "bert_large_cased_en": {'base': "roberta.base",
                            'base_model': "bert-large-cased",
                            'MODEL_TYPE': "bert_large_en_cased",
                            'MODEL_SUFFIX': "cased",
                            'MODEL_SPEC_STR': "2018_10_18/cased_L-12_H-768_A-12",
                            'MODEL_SPEC_STR_DIR': "cased_L-12_H-768_A-12/",
                            'VOCAB_SIZE': 28996,
                            'NUM_LAYERS': 24,
                            'NUM_ATTN_HEADS': 16,
                            'EMBEDDING_SIZE': 1024},
    "bert_large_uncased_en": {'base': "roberta.base",
                              'base_model': "bert-large-uncased",
                              'MODEL_TYPE': "bert_large_en_uncased",
                              'MODEL_SUFFIX': "uncased",
                              'MODEL_SPEC_STR': "2020_02_20/uncased_L-12_H-768_A-12",
                              'MODEL_SPEC_STR_DIR': "uncased_L-12_H-768_A-12/",
                              'VOCAB_SIZE': 30522,
                              'NUM_LAYERS': 24,
                              'NUM_ATTN_HEADS': 16,
                              'EMBEDDING_SIZE': 768
                              },
    "bert_medium_uncased_en": {'base': "roberta.base",
                               'base_model': "bert-base-uncased",
                               'MODEL_TYPE': "bert_medium_en_uncased",
                               'MODEL_SUFFIX': "uncased",
                               'MODEL_SPEC_STR': "2020_02_20/uncased_L-8_H-512_A-8",
                               'MODEL_SPEC_STR_DIR': "",
                               'VOCAB_SIZE': 30522,
                               'NUM_LAYERS': 8,
                               'NUM_ATTN_HEADS': 8,
                               'EMBEDDING_SIZE': 512
                               },
    "bert_small_uncased_en": {'base': "roberta.base",
                              'base_model': "bert-base-uncased",
                              'MODEL_TYPE': "bert_small_en_uncased",
                              'MODEL_SUFFIX': "uncased",
                              'MODEL_SPEC_STR': "2020_02_20/uncased_L-4_H-512_A-8",
                              'MODEL_SPEC_STR_DIR': "",
                              'VOCAB_SIZE': 30522,
                              'NUM_LAYERS': 4,
                              'NUM_ATTN_HEADS': 8,
                              'EMBEDDING_SIZE': 768},
    "bert_base_uncased": {'base': "roberta.base",
                          'base_model': "bert-base-uncased",
                          'MODEL_TYPE': "bert_tiny_en_uncased",
                          'MODEL_SUFFIX': "uncased",
                          'MODEL_SPEC_STR': "2018_10_18/uncased_L-12_H-768_A-12",
                          'MODEL_SPEC_STR_DIR': "uncased_L-12_H-768_A-12",
                          'VOCAB_SIZE': 30522,
                          'NUM_LAYERS': 2,
                          'NUM_ATTN_HEADS': 2,
                          'EMBEDDING_SIZE': 128}
}

flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def download_model(preset):
    print("-> Download original weights in " + preset)
    zip_path = f"""https://storage.googleapis.com/bert_models/{PRESET_MAP[preset]['MODEL_SPEC_STR']}.zip"""  # 768
    path_file = keras.utils.get_file(
        f"""{preset}.zip""",
        zip_path,
        extract=False,
        archive_format="tar",
    )
    extract_dir = f"./content/{preset}/{PRESET_MAP[preset]['MODEL_SPEC_STR_DIR']}"
    vocab_path = os.path.join(extract_dir, "vocab.txt")
    checkpoint_path = os.path.join(extract_dir, "bert_model.ckpt")
    config_path = os.path.join(extract_dir, "bert_config.json")
    os.system(f"unzip -o -d ./content/{preset}  {path_file}")

    return vocab_path, checkpoint_path, config_path


def convert_checkpoints(preset, checkpoint_path, config_dict):
    print("\n-> Convert original weights to KerasNLP format.")

    # Transformer layers.
    vars = tf.train.list_variables(checkpoint_path)
    weights = {}
    for name, shape in vars:
        weight = tf.train.load_variable(checkpoint_path, name)
        weights[name] = weight

    model = keras_nlp.models.BertBackbone.from_preset(config_dict['MODEL_TYPE'],
                                                      load_weights=True)
    if preset in ['bert_base_en']:
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
            if preset == 'bert_base_uncased':
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
        if preset == 'bert_base_uncased':
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
                    (config_dict['EMBEDDING_SIZE'], config_dict['NUM_ATTN_HEADS'], -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._key_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/key/bias"].reshape(
                    (config_dict['NUM_ATTN_HEADS'], -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.kernel.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/query/kernel"].reshape(
                    (config_dict['EMBEDDING_SIZE'], config_dict['NUM_ATTN_HEADS'], -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._query_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/query/bias"].reshape(
                    (config_dict['NUM_ATTN_HEADS'], -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.kernel.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/value/kernel"].reshape(
                    (config_dict['EMBEDDING_SIZE'], config_dict['NUM_ATTN_HEADS'], -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._value_dense.bias.assign(
                weights[f"bert/encoder/layer_{i}/attention/self/value/bias"].reshape(
                    (config_dict['NUM_ATTN_HEADS'], -1)
                )
            )
            model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._output_dense.kernel.assign(
                weights[
                    f"bert/encoder/layer_{i}/attention/output/dense/kernel"
                ].reshape((config_dict['NUM_ATTN_HEADS'], -1, config_dict['EMBEDDING_SIZE']))
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
    print(f"\n-> Save KerasNLP model weights to `{config_dict['base_model']}.h5`.")
    model.save_weights(f"{config_dict['base_model']}.h5")

    return model


def define_preprocessor(vocab_path, checkpoint_path, config_path, model):
    def preprocess(x):
        tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=vocab_path, lowercase=False
        )

        return keras_nlp.models.BertPreprocessor(tokenizer(x))

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
        keras_nlp_preprocessor,
        keras_nlp_model,
        hf_tokenizer,
        hf_model,
):
    print("\n-> Check the outputs.")
    sample_text = ["cricket is awesome, easily the best sport in the world!"]

    # KerasNLP
    keras_nlp_inputs = keras_nlp_preprocessor(tf.constant(sample_text))
    keras_nlp_output = keras_nlp_model.predict(keras_nlp_inputs)[
        "sequence_output"
    ]

    # HF
    hf_inputs = hf_tokenizer(
        sample_text, padding="max_length", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasNLP output:", keras_nlp_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print('keras_nlp_output', keras_nlp_output.shape)
    print('hf_output', hf_output.detach().numpy().shape)
    print("Difference:", np.mean(keras_nlp_output - hf_output.detach().numpy()))


def extract_vocab(hf_tokenizer, vocab_path):
    spm_path = os.path.join('models/' + FLAGS.preset, "spiece.model")
    print(f"\n-> Save KerasNLP SPM vocabulary file to `{spm_path}`.")

    os.makedirs('models/' + FLAGS.preset, exist_ok=True)
    shutil.copyfile(
        transformers.utils.hub.get_file_from_repo("bert-base-uncased", "tokenizer_config.json"

                                                  ),
        spm_path,
    )
    keras_nlp_tokenizer = keras_nlp.models.BertTokenizer(

        vocabulary=vocab_path
    )
    keras_nlp_preprocessor = keras_nlp.models.BertPreprocessor(
        keras_nlp_tokenizer
    )

    print("-> Print MD5 checksum of the vocab files.")
    print(f"`{spm_path}` md5sum: ", get_md5_checksum(spm_path))

    return keras_nlp_preprocessor


def main(_):
    assert (
            FLAGS.preset in PRESET_MAP.keys()
    ), f'Invalid preset {FLAGS.preset}. Must be one of {",".join(PRESET_MAP.keys())}'

    vocab_path, checkpoint_path, config_path = download_model(FLAGS.preset)

    keras_nlp_model = convert_checkpoints(FLAGS.preset, checkpoint_path, PRESET_MAP[FLAGS.preset])

    hf_model = transformers.AutoModel.from_pretrained(PRESET_MAP[FLAGS.preset]['base_model'])
    hf_model.eval()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(PRESET_MAP[FLAGS.preset]['base_model'])
    keras_nlp_preprocessor = extract_vocab(hf_tokenizer, vocab_path)
    check_output(
        keras_nlp_preprocessor,
        keras_nlp_model,
        hf_tokenizer,
        hf_model,
    )


if __name__ == "__main__":
    app.run(main)
