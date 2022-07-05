# Copyright 2022 The KerasNLP Authors
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
"""Utility functions for writing training scripts."""

import glob
import os


def list_filenames_for_arg(arg_pattern):
    """List filenames from a comma separated list of files, dirs, and globs."""
    input_filenames = []
    for pattern in arg_pattern.split(","):
        pattern = os.path.expanduser(pattern)
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "**", "*")
        for filename in glob.iglob(pattern, recursive=True):
            if not os.path.isdir(filename):
                input_filenames.append(filename)
    return input_filenames
