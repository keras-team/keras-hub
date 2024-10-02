import os
import shutil

import numpy as np
import tensorflow as tf
import transformers
from absl import app
from absl import flags
from checkpoint_conversion_utils import get_md5_checksum

import keras_hub

PRESET_MAP = {
    "f_net_base_en": "google/fnet-base",
    "f_net_large_en": "google/fnet-large",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def convert_checkpoints(hf_model):
    print("\n-> Convert original weights to KerasHub format.")

    print("\n-> Load KerasHub model.")
    keras_hub_model = keras_hub.models.FNetBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(list(hf_wts.keys()))

    keras_hub_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["embeddings.word_embeddings.weight"]
    )
    keras_hub_model.get_layer("position_embedding").position_embeddings.assign(
        hf_wts["embeddings.position_embeddings.weight"]
    )
    keras_hub_model.get_layer("segment_embedding").embeddings.assign(
        hf_wts["embeddings.token_type_embeddings.weight"]
    )

    keras_hub_model.get_layer("embeddings_layer_norm").gamma.assign(
        hf_wts["embeddings.LayerNorm.weight"]
    )
    keras_hub_model.get_layer("embeddings_layer_norm").beta.assign(
        hf_wts["embeddings.LayerNorm.bias"]
    )

    keras_hub_model.get_layer("embedding_projection").kernel.assign(
        hf_wts["embeddings.projection.weight"].T
    )
    keras_hub_model.get_layer("embedding_projection").bias.assign(
        hf_wts["embeddings.projection.bias"]
    )

    for i in range(keras_hub_model.num_layers):
        keras_hub_model.get_layer(
            f"f_net_layer_{i}"
        )._mixing_layer_norm.gamma.assign(
            hf_wts[f"encoder.layer.{i}.fourier.output.LayerNorm.weight"].numpy()
        )
        keras_hub_model.get_layer(
            f"f_net_layer_{i}"
        )._mixing_layer_norm.beta.assign(
            hf_wts[f"encoder.layer.{i}.fourier.output.LayerNorm.bias"].numpy()
        )

        keras_hub_model.get_layer(
            f"f_net_layer_{i}"
        )._intermediate_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_hub_model.get_layer(
            f"f_net_layer_{i}"
        )._intermediate_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
        )

        keras_hub_model.get_layer(
            f"f_net_layer_{i}"
        )._output_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_hub_model.get_layer(f"f_net_layer_{i}")._output_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.bias"].numpy()
        )

        keras_hub_model.get_layer(
            f"f_net_layer_{i}"
        )._output_layer_norm.gamma.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
        )
        keras_hub_model.get_layer(
            f"f_net_layer_{i}"
        )._output_layer_norm.beta.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
        )

    keras_hub_model.get_layer("pooled_dense").kernel.assign(
        hf_wts["pooler.dense.weight"].transpose(1, 0).numpy()
    )
    keras_hub_model.get_layer("pooled_dense").bias.assign(
        hf_wts["pooler.dense.bias"].numpy()
    )

    # Save the model.
    print("\n-> Save KerasHub model weights.")
    keras_hub_model.save_weights(os.path.join(FLAGS.preset, "model.h5"))

    return keras_hub_model


def extract_vocab(hf_tokenizer):
    spm_path = os.path.join(FLAGS.preset, "spiece.model")
    print(f"\n-> Save KerasHub SPM vocabulary file to `{spm_path}`.")

    shutil.copyfile(
        transformers.utils.hub.get_file_from_repo(
            hf_tokenizer.name_or_path, "spiece.model"
        ),
        spm_path,
    )

    keras_hub_tokenizer = keras_hub.models.FNetTokenizer(
        proto=spm_path,
    )
    keras_hub_preprocessor = keras_hub.models.FNetTextClassifierPreprocessor(
        keras_hub_tokenizer
    )

    print("-> Print MD5 checksum of the vocab files.")
    print(f"`{spm_path}` md5sum: ", get_md5_checksum(spm_path))

    return keras_hub_preprocessor


def check_output(
    keras_hub_preprocessor,
    keras_hub_model,
    hf_tokenizer,
    hf_model,
):
    print("\n-> Check the outputs.")
    sample_text = ["cricket is awesome, easily the best sport in the world!"]

    # KerasHub
    keras_hub_inputs = keras_hub_preprocessor(tf.constant(sample_text))
    keras_hub_output = keras_hub_model.predict(keras_hub_inputs)[
        "sequence_output"
    ]

    # HF
    hf_inputs = hf_tokenizer(
        sample_text, padding="max_length", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasHub output:", keras_hub_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print("Difference:", np.mean(keras_hub_output - hf_output.detach().numpy()))

    # Show the MD5 checksum of the model weights.
    print(
        "Model md5sum: ",
        get_md5_checksum(os.path.join(FLAGS.preset, "model.h5")),
    )


def main(_):
    os.makedirs(FLAGS.preset)

    hf_model_name = PRESET_MAP[FLAGS.preset]

    print("\n-> Load HF model and HF tokenizer.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    keras_hub_model = convert_checkpoints(hf_model)
    print("\n -> Load KerasHub preprocessor.")
    keras_hub_preprocessor = extract_vocab(hf_tokenizer)

    check_output(
        keras_hub_preprocessor,
        keras_hub_model,
        hf_tokenizer,
        hf_model,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
