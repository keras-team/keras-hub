import ast
import pathlib

from keras_hub.src.tests.test_case import TestCase

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
# Test helpers may use transformers, and the vLLM plugin runs inside it.
EXCLUDED_FOLDERS = ("tests", "vllm")


def imports_transformers(node):
    """Returns whether an `ast` node imports `transformers`.

    Covers import statements and `importlib.import_module` or `__import__`
    calls whose first argument is a string literal.
    """
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.level == 0:
        names = [node.module]
    elif (
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None))
        in ("import_module", "__import__")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        names = [node.args[0].value]
    else:
        return False
    return any(name.split(".")[0] == "transformers" for name in names)


class TransformersImportTest(TestCase):
    def test_library_code_does_not_import_transformers(self):
        # The reference parity tests compare keras-hub models with
        # transformers. That comparison is only meaningful if keras-hub
        # computes its outputs itself, so only tests may use transformers.
        paths = [
            path
            for path in sorted(SRC_ROOT.rglob("*.py"))
            if not path.name.endswith("_test.py")
            and path.relative_to(SRC_ROOT).parts[0] not in EXCLUDED_FOLDERS
        ]
        self.assertNotEqual(paths, [])
        offenders = []
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if imports_transformers(node):
                    relative = path.relative_to(SRC_ROOT.parents[1])
                    offenders.append(f"{relative}:{node.lineno}")

        self.assertEqual(
            offenders,
            [],
            "Library code must not import `transformers`, at module level or "
            "inside a function: a port must compute its outputs itself, or "
            "the reference parity tests compare transformers with itself. "
            f"Only tests may import it: {offenders}",
        )

    def test_imports_transformers(self):
        def found(source):
            return any(
                imports_transformers(node)
                for node in ast.walk(ast.parse(source))
            )

        for source in (
            "import transformers",
            "import transformers.models.bert as bert",
            "from transformers import BertModel",
            "def f():\n    from transformers.utils import logging",
            "importlib.import_module('transformers')",
            "import_module('transformers.models')",
            "__import__('transformers')",
        ):
            self.assertTrue(found(source), source)
        for source in (
            "from keras_hub.src.utils.transformers import convert_bert",
            "from .transformers import preset_loader",
            "import transformers_helper",
            "importlib.import_module(name)",
        ):
            self.assertFalse(found(source), source)
