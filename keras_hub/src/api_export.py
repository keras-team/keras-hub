# Copyright 2024 The KerasHub Authors
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

import types

from keras.saving import register_keras_serializable

try:
    import namex
except ImportError:
    namex = None


def maybe_register_serializable(path, symbol):
    if isinstance(path, (list, tuple)):
        # If we have multiple export names, actually make sure to register these
        # first. This makes sure we have a backward compat mapping of old
        # serialized names to new class.
        for name in path:
            name = name.split(".")[-1]
            register_keras_serializable(package="keras_nlp", name=name)(symbol)
            register_keras_serializable(package="keras_hub", name=name)(symbol)
    if isinstance(symbol, types.FunctionType) or hasattr(symbol, "get_config"):
        # We register twice, first with keras_nlp, second with keras_hub,
        # so loading still works for classes saved as "keras_nlp".
        register_keras_serializable(package="keras_nlp")(symbol)
        register_keras_serializable(package="keras_hub")(symbol)


if namex:

    class keras_hub_export(namex.export):
        def __init__(self, path):
            super().__init__(package="keras_hub", path=path)

        def __call__(self, symbol):
            maybe_register_serializable(self.path, symbol)
            return super().__call__(symbol)

else:

    class keras_hub_export:
        def __init__(self, path):
            self.path = path

        def __call__(self, symbol):
            maybe_register_serializable(self.path, symbol)
            return symbol
