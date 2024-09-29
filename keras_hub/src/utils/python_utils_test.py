from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.python_utils import classproperty


class ClassPropertyTest(TestCase):
    def test_class_property(self):
        class Foo:
            @classproperty
            def bar(cls):
                return "class property"

        self.assertAllEqual(Foo.bar, "class property")
