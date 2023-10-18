# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import tarfile
import zipfile
import inspect

import keras as keraslib
from keras.engine.base_layer_utils import make_variable


def get_md5_checksum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def extract_files_from_archive(archive_file_path):
    if archive_file_path.endswith(".tar.gz"):
        with tarfile.open(archive_file_path, "r:gz") as tar:
            return tar.extractall()
    elif archive_file_path.endswith(".zip"):
        with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
            return zip_ref.extractall()


def port_weights_by_creation_order(source_model_fn, dest_model_fn, debug=False):
    """Assign weights between models by intercepting variable creation.
    For each model makes a flat list of all variables created, in order."""
    ALL_VARS = []

    def make_var(name, shape=None, **kwargs):
        if debug:
            stack = inspect.stack()
            instance = stack[1][0].f_locals["self"]
            cls = instance.__class__.__name__
            print(f"Class {cls} creating {name} with shape {shape}.")

        v = make_variable(name, shape=shape, **kwargs)
        ALL_VARS.append(v)
        return v

    # Patch make variable.
    keraslib.engine.base_layer_utils.make_variable = make_var

    source_model = source_model_fn()
    source_model_vars = ALL_VARS[:]

    [ALL_VARS.pop(0) for _ in list(ALL_VARS)]
    dest_model = dest_model_fn()
    if len(ALL_VARS) != len(source_model_vars):
        raise ValueError(
            f"Variable counts do not match. 1st model: {len(source_model_vars)} "
            "vars, 2nd model: {len(ALL_VARS)} vars"
        )
    for v1, v2 in zip(source_model_vars, ALL_VARS):
        v2.assign(v1.numpy())

    # Unpatch make variable.
    keraslib.engine.base_layer_utils.make_variable = make_variable

    return source_model, dest_model