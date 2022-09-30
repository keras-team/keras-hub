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

from collections import defaultdict

checkpoints = {}


# Metadata for loading pretrained tokenizer vocabularies.
# TODO: Vocabularies of DistilBERT are the same as BERT's, including the
# hash. We can have a common folder for both BERT and DistilBERT vocabularies.
vocabularies = {}

# Index checkpoints by arch compatibility and create lookup function
checkpoints_per_arch = defaultdict(set)
for arch, metadata in checkpoints.items():
    checkpoints_per_arch[metadata["model"]].add(arch)


def compatible_checkpoints(arch):
    """Returns a list of compatible checkpoints per arch"""
    return checkpoints_per_arch[arch]
