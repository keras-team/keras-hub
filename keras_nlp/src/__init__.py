# Copyright 2021 The KerasNLP Authors
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

# sentencepiece is segfaulting on tf-nightly if tensorflow is imported first.
# This is a temporary fix to restore our nightly testing to green, while we look
# for a longer term solution.
try:
    import sentencepiece
except ImportError:
    pass

from keras_nlp.src import layers
from keras_nlp.src import metrics
from keras_nlp.src import models
from keras_nlp.src import samplers
from keras_nlp.src import tokenizers
from keras_nlp.src import utils
from keras_nlp.src.utils.preset_utils import upload_preset
from keras_nlp.src.version_utils import __version__
from keras_nlp.src.version_utils import version
