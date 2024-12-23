"""Update all preset files to use the latest version on kaggle.

Run from the base of the repo.

Usage:
```
python tools/admin/update_all_versions.py
```
"""

import os
import pathlib

import kagglehub

import keras_hub


def update():
    presets = keras_hub.models.Backbone.presets
    for preset in sorted(presets.keys()):
        uri = presets[preset]["kaggle_handle"]
        kaggle_handle = uri.removeprefix("kaggle://")
        old_version = os.path.basename(kaggle_handle)
        kaggle_handle = os.path.dirname(kaggle_handle)
        hub_dir = kagglehub.model_download(kaggle_handle, path="metadata.json")
        new_version = os.path.basename(os.path.dirname(hub_dir))
        if old_version != new_version:
            print(f"Updating {preset} from {old_version} to {new_version}")
            for path in pathlib.Path(".").glob("keras_hub/**/*_presets.py"):
                with open(path, "r") as file:
                    contents = file.read()
                new_uri = os.path.dirname(uri) + f"/{new_version}"
                contents = contents.replace(f'"{uri}"', f'"{new_uri}"')
                with open(path, "w") as file:
                    file.write(contents)


if __name__ == "__main__":
    update()
