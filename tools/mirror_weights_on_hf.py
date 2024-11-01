import json
import os
import re
import shutil

from huggingface_hub import HfApi
from kaggle.api.kaggle_api_extended import KaggleApi

import keras_hub
import keras_hub.src.utils.preset_utils as utils

try:
    import kagglehub
except ImportError:
    kagglehub = None

HF_BASE_URI = "hf://keras"
JSON_FILE_PATH = "tools/hf_uploaded_presets.json"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


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
            model_variant = kaggle_handle_path.split("/")[3]
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


def update_model_cards_on_hugging_face(presets):
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    for model, data in presets.items():
        try:
            kaggle_handle = data["kaggle_handle"].removeprefix("kaggle://")
            owner = "keras"
            model_slug = kaggle_handle.split("/")[1]
            model_metadata = kaggle_api.get_model_with_http_info(
                owner, model_slug
            )
            description = model_metadata[0]["description"]
            usage = model_metadata[0]["instances"][0]["usage"].replace(
                "${VARIATION_SLUG}", model
            )
            usage = re.sub(
                r'\.from_preset\(".*?"\)', f'.from_preset("{model}")', usage
            )
            hf_usage = usage.replace(model, f"hf://keras/{model}")

            print(f"Downloading model metadata from Kaggle: {model}")

            # --- Construct Model Card Markup ---
            model_card_markup = (
                "---\nlibrary_name: keras-hub\n---\n"
                + f"### Model Overview\n{description}\n\n"
            )

            # Add usage sections if `usage` is not empty
            if usage:
                model_card_markup += (
                    f"### Example Usage\n{usage}\n\n"
                    "## Example Usage with Hugging Face URI\n\n"
                    f"{hf_usage}\n"
                )

            model_card_markup = (
                model_card_markup.replace("keras-nlp", "keras-hub")
                .replace("keras_nlp", "keras_hub")
                .replace("KerasNLP", "KerasHub")
                .replace("&gt;=", ">=")
            )

            # --- Save Model Card Content to README.md ---
            readme_path = "README.md"
            with open(readme_path, "w") as readme_file:
                readme_file.write(model_card_markup)

            # --- Hugging Face API Authentication and README Upload ---
            hf_api = HfApi()
            repo_id = f"keras/{model}"

            # Upload README.md to Hugging Face repository
            hf_api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                token=HF_TOKEN,
                commit_message="Update README.md with new model card content",
            )
            print(f"Uploaded README.md to Hugging Face repository: {repo_id}")

            # --- Clean up the README.md file after upload ---
            os.remove(readme_path)
            print(f"Deleted local README.md for {model}")

        except Exception as e:
            print(f"Error updating model card for {model}: {e}")
            continue


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

    # Step 6: Update HuggingFace model card
    print("Updating model cards on HuggingFace")
    update_model_cards_on_hugging_face(presets)
    print("Updating model cards on HuggingFace is done")


if __name__ == "__main__":
    main()
