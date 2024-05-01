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

from keras_nlp.src.tests.test_case import TestCase
from keras_nlp.src.utils.python_utils import classproperty


class ClassPropertyTest(TestCase):
    def test_class_property(self):
        class Foo:
            @classproperty
            def bar(cls):
                return "class property"

        self.assertAllEqual(Foo.bar, "class property")
