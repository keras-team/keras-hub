import ast
import pathlib

from keras_hub.src.tests.test_case import TestCase

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
OPTIONAL_PACKAGES = ("tensorflow", "tensorflow_text")


def _imported_packages(node):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        return []
    return [name.split(".")[0] for name in names]


class ImportTest(TestCase):
    def test_no_module_level_optional_imports(self):
        # `import keras_hub` must work without TensorFlow installed, so an
        # optional package may only be imported inside the function that needs
        # it, or behind a `try` that falls back to `None`. Tests are exempt.
        offenders = []
        for path in sorted(SRC_ROOT.rglob("*.py")):
            if path.name.endswith("_test.py"):
                continue
            if "tests" in path.relative_to(SRC_ROOT).parts:
                continue
            source = path.read_text(encoding="utf-8")
            for node in ast.parse(source).body:
                packages = _imported_packages(node)
                if any(p in OPTIONAL_PACKAGES for p in packages):
                    relative = path.relative_to(SRC_ROOT.parent)
                    offenders.append(f"{relative}:{node.lineno}")

        self.assertEqual(
            offenders,
            [],
            "Wrap these imports in `try: ... except ImportError: ... = None`, "
            f"or move them inside the function that needs them: {offenders}",
        )
