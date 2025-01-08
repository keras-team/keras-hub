"""Update all json files for all models on Kaggle.

Run tools/admin/update_all_versions.py before running this tool to make sure
all our kaggle links point to the latest version!

This script downloads all models from KaggleHub, loads and re-serializes all
json files, and reuploads them. This can be useful when changing our metadata or
updating our saved configs.

This script relies on private imports from preset_utils and may need updates
when it is re-run.

Usage:
```
# Preview changes.
python tools/admin/update_all_json.py
# Upload changes.
python tools/admin/update_all_json.py --upload
# Resume after a failure.
python tools/admin/update_all_json.py --upload --start_at=gemma_2b_en
```
"""

import difflib
import os
import pathlib
import shutil

import kagglehub
import torch
from absl import app
from absl import flags

os.environ["KERAS_BACKEND"] = "torch"

import keras_hub
from keras_hub.src.utils import preset_utils

FLAGS = flags.FLAGS
flags.DEFINE_boolean("upload", False, "Upload updated models.")
flags.DEFINE_string("start_at", "", "Resume at given preset.")


BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def diff(in_path, out_path):
    with open(in_path) as in_file, open(out_path) as out_file:
        in_lines = in_file.readlines()
        out_lines = out_file.readlines()
        # Ignore updates to upload_date.
        if "metadata.json" in in_path.name:
            in_lines = [line for line in in_lines if "date" not in line]
            out_lines = [line for line in out_lines if "date" not in line]
        diff = difflib.unified_diff(
            in_lines,
            out_lines,
        )
        diff = list(diff)
        if not diff:
            return False
        for line in diff:
            if line.startswith("+"):
                print("    " + GREEN + line + RESET, end="")
            elif line.startswith("-"):
                print("    " + RED + line + RESET, end="")
            else:
                print("    " + line, end="")
        print()
        return True


def main(argv):
    presets = keras_hub.models.Backbone.presets
    output_parent = pathlib.Path("updates")
    output_parent.mkdir(parents=True, exist_ok=True)

    remaining = sorted(presets.keys())
    if FLAGS.start_at:
        remaining = remaining[remaining.index(FLAGS.start_at) :]

    for preset in remaining:
        handle = presets[preset]["kaggle_handle"].removeprefix("kaggle://")
        handle_no_version = os.path.dirname(handle)
        builtin_name = os.path.basename(handle_no_version)

        # Download the full model with KaggleHub.
        input_dir = kagglehub.model_download(handle)
        input_dir = pathlib.Path(input_dir)
        output_dir = output_parent / builtin_name
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(input_dir, output_dir)

        # Manually create saver/loader objects.
        config = preset_utils.load_json(preset, preset_utils.CONFIG_FILE)
        loader = preset_utils.KerasPresetLoader(preset, config)
        saver = preset_utils.KerasPresetSaver(output_dir)

        # Update all json files.
        print(BOLD + handle + RESET)
        updated = False
        for file in input_dir.glob("*.json"):
            if file.name == preset_utils.METADATA_FILE:
                # metadata.json is handled concurrently with config.json.
                continue
            config = preset_utils.load_json(preset, file.name)
            layer = loader._load_serialized_object(config)
            saver._save_serialized_object(layer, file.name)
            if file.name == preset_utils.CONFIG_FILE:
                # Handle metadata.json with config.json.
                print("  " + BOLD + preset_utils.METADATA_FILE + RESET)
                saver._save_metadata(layer)
                name = preset_utils.METADATA_FILE
                if diff(input_dir / name, output_dir / name):
                    updated = True
            print("  " + BOLD + file.name + RESET)
            if diff(input_dir / file.name, output_dir / file.name):
                updated = True
            del layer

        if not updated:
            continue

        # Reupload the model if any json files were updated.
        if FLAGS.upload:
            print(BOLD + "Uploading " + handle_no_version + RESET)
            kagglehub.model_upload(
                handle_no_version,
                output_dir,
                version_notes="updated json files",
            )
        else:
            print(BOLD + "Preview. Not uploading " + handle_no_version + RESET)

    if FLAGS.upload:
        print(BOLD + "Wait a few hours (for kaggle to process models)." + RESET)
        print(BOLD + "Then run tasks/admin/update_all_versions.py" + RESET)


if __name__ == "__main__":
    with torch.device("meta"):
        app.run(main)
