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

import tensorflow as tf

from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


class ClassWithClassProperty:
    @classproperty
    def foo(cls):
        return "class property"


@format_docstring(adjective="salubrious")
def foo():
    """It was a {{adjective}} November day."""
    return "function"


@format_docstring(adjective="smelly", name="Mortimer")
class ClassWithDocstrings:
    """I saw my {{adjective}} friend {{name}}."""

    def __init__(self):
        self.bar = "property"

    @classmethod
    @format_docstring(noun="cactus", bodypart="nostril")
    def foo(cls):
        """He was holding a {{noun}} in his {{bodypart}}."""
        return "class method"


@format_docstring(nickname="dumdum")
def bar():
    """Use `{}` to create a dictionary, {{nickname}}."""
    return "function"


class ClassPropertyTest(tf.test.TestCase):
    def test_class_property(self):
        self.assertAllEqual(ClassWithClassProperty.foo, "class property")


class FormatDocstringTest(tf.test.TestCase):
    def test_function(self):
        self.assertAllEqual(foo(), "function")
        self.assertAllEqual(foo.__doc__, "It was a salubrious November day.")

    def test_class(self):
        self.assertAllEqual(ClassWithDocstrings().bar, "property")
        self.assertAllEqual(
            ClassWithDocstrings.__doc__, "I saw my smelly friend Mortimer."
        )

    def test_class_method(self):
        self.assertAllEqual(ClassWithDocstrings.foo(), "class method")
        self.assertAllEqual(
            ClassWithDocstrings.foo.__doc__,
            "He was holding a cactus in his nostril.",
        )
        self.assertAllEqual(
            ClassWithDocstrings.foo.__func__.__doc__,
            "He was holding a cactus in his nostril.",
        )

    def test_brackets(self):
        self.assertAllEqual(bar(), "function")
        self.assertAllEqual(
            bar.__doc__, "Use `{}` to create a dictionary, dumdum."
        )
