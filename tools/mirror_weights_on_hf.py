import json
import shutil

import keras_hub
import keras_hub.src.utils.preset_utils as utils

try:
    import kagglehub
except ImportError:
    kagglehub = None

HF_BASE_URI = "hf://keras"
JSON_FILE_PATH = "tools/hf_uploaded_presets.json"


def load_latest_hf_uploads(json_file_path):
    # Load the latest HF uploads from JSON
    with open(json_file_path, "r") as json_file:
        latest_hf_uploads = set(json.load(json_file))
    print("Loaded latest HF uploads from JSON file.")
    return latest_hf_uploads


def download_and_upload_missing_models(missing_in_hf_uploads):
    uploaded_handles = []
    errored_uploads = []
    for kaggle_handle in missing_in_hf_uploads:
        try:
            kaggle_handle_path = kaggle_handle.removeprefix("kaggle://")
            model_variant = kaggle_handle.split("/")[3]
            hf_uri = f"{HF_BASE_URI}/{model_variant}"
            # Skip Gemma models
            if "gemma" in kaggle_handle_path:
                print(f"Skipping Gemma model preset: {kaggle_handle_path}")
                continue

            print(f"Downloading model: {kaggle_handle_path}")
            model_file_path = kagglehub.model_download(kaggle_handle_path)

            print(f"Uploading to HF: {hf_uri}")
            keras_hub.upload_preset(hf_uri, model_file_path)

            print(f"Cleaning up: {model_file_path}")
            shutil.rmtree(model_file_path)

            # Add to the list of successfully uploaded handles
            uploaded_handles.append(kaggle_handle)
        except Exception as e:
            print(
                f"Error in downloading  and uploading preset {kaggle_handle}: {e}"
            )
            errored_uploads.append(kaggle_handle)

    print("All missing models processed.")
    return uploaded_handles, errored_uploads


def update_hf_uploads_json(json_file_path, latest_kaggle_handles):
    with open(json_file_path, "w") as json_file:
        json.dump(latest_kaggle_handles, json_file, indent=4)

    print("Updated hf_uploaded_presets.json with newly uploaded handles.")


def main():
    print("Starting the model presets mirroring on HF")

    # Step 1: Load presets
    presets = utils.BUILTIN_PRESETS
    print("Loaded presets from utils.")

    # Step 2: Load latest HF uploads
    latest_hf_uploads = load_latest_hf_uploads(JSON_FILE_PATH)

    # Step 3: Find missing uploads
    latest_kaggle_handles = {
        data["kaggle_handle"] for model, data in presets.items()
    }
    missing_in_hf_uploads = latest_kaggle_handles - latest_hf_uploads
    print(f"Found {len(missing_in_hf_uploads)} models missing on HF.")

    # Step 4: Download and upload missing models
    _, errored_uploads = download_and_upload_missing_models(
        missing_in_hf_uploads
    )

    # Step 5: Update JSON file with newly uploaded handles
    update_hf_uploads_json(
        JSON_FILE_PATH, list(set(latest_kaggle_handles) - set(errored_uploads))
    )
    print("uploads for the following models failed: ", errored_uploads)
    print("Rest of the models up to date on HuggingFace")


if __name__ == "__main__":
    main()
