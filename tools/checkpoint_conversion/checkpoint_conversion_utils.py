import hashlib
import os
import shutil
import tarfile
import zipfile

import numpy as np
from keras import ops


def get_md5_checksum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def _is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(os.path.join(abs_directory, target))
    try:
        return os.path.commonpath([abs_directory, abs_target]) == abs_directory
    except ValueError:
        return False


def extract_files_from_archive(archive_file_path):
    if archive_file_path.endswith(".tar.gz"):
        with tarfile.open(archive_file_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not _is_within_directory(".", member.name):
                    raise Exception(
                        f"Attempted Path Traversal in Tar File: {member.name}"
                    )
            return tar.extractall()
    elif archive_file_path.endswith(".zip"):
        with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                if not _is_within_directory(".", member.filename):
                    raise Exception(
                        "Attempted Path Traversal in Zip File: "
                        f"{member.filename}"
                    )
            return zip_ref.extractall()


def download_gcs_file(gcs_uri, destination_file_name):
    """Downloads a file from Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        print(
            "The `google-cloud-storage` package is not installed. "
            "Please install it with `pip install google-cloud-storage` "
            "to download files from Google Cloud Storage."
        )

    storage_client = storage.Client.create_anonymous_client()
    bucket_name = gcs_uri.split("/")[2]
    source_blob_name = "/".join(gcs_uri.split("/")[3:])

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file
    blob.download_to_filename(destination_file_name)


def extract_response(text):
    """Strip the prompt/chat-template prefix from generated text."""
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text.strip()


def count_keras_params(backbone):
    """Count unique parameters (handles tied weights)."""
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(ops.convert_to_numpy(w).size for w in unique)


def test_parameter_count(keras_backbone, hf_param_count):
    print("\n" + "=" * 50)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 50)
    keras_params = count_keras_params(keras_backbone)
    print(f"\n  KerasHub params: {keras_params:,}")
    print(f"  HF params:       {hf_param_count:,}")
    assert keras_params == hf_param_count, (
        f"Parameter count mismatch: keras={keras_params}, hf={hf_param_count}"
    )
    print("  Parameter counts match.")


def save_preset(keras_model, preset_name):
    """Save the model to ``./<preset_name>`` and return the path."""
    preset_path = f"./{preset_name}"
    if os.path.exists(preset_path):
        print(f"\n-> Removing existing preset at {preset_path}...")
        shutil.rmtree(preset_path)
    print(f"-> Saving KerasHub preset to {preset_path}...")
    keras_model.save_to_preset(preset_path)
    print(f"  Preset saved to {preset_path}")
    return preset_path


def validate_reloaded_preset(model_cls, preset_path):
    """Reload the saved artifact and run a forward sanity check."""
    print("\n" + "=" * 50)
    print("SAVE/RELOAD VALIDATION")
    print("=" * 50)
    reloaded = model_cls.from_preset(preset_path)
    preprocessed = reloaded.preprocessor.generate_preprocess(
        ["What is the capital of France?"], sequence_length=8
    )
    logits = ops.convert_to_numpy(reloaded(preprocessed))
    assert np.isfinite(logits).all(), (
        "reloaded model produced non-finite logits"
    )
    print("  Reloaded preset produces finite logits.")
