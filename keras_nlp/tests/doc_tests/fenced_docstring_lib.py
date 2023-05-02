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

"""Fenced docstring docstest lib for KerasNLP."""

import ast
import doctest
import re
import textwrap
from typing import List

try:
    import astor
except:
    astor = None


class FencedCellParser(doctest.DocTestParser):
    """Implements test parsing for ``` fenced cells.

    https://docs.python.org/3/library/doctest.html#doctestparser-objects

    The `get_examples` method receives a string and returns an
    iterable of `doctest.Example` objects.

    Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docs/fenced_doctest_lib.py.
    """

    patched = False

    def __init__(self, fence_label="python"):
        super().__init__()

        if not self.patched:
            # The default doctest compiles in "single" mode. The fenced block may
            # contain multiple statements. The `_patch_compile` function fixes the
            # compile mode.
            doctest.compile = _patch_compile
            print(
                textwrap.dedent(
                    """
          *********************************************************************
          * Caution: `fenced_doctest` patches `doctest.compile` don't use this
          *   in the same binary as any other doctests.
          *********************************************************************
          """
                )
            )
            type(self).patched = True

        # Match anything, except if the look-behind sees a closing fence.
        no_fence = "(.(?<!```))*?"
        self.fence_cell_re = re.compile(
            rf"""
        ^(                             # After a newline
            \s*```\s*({fence_label})\n   # Open a labeled ``` fence
            (?P<doctest>{no_fence})      # Match anything except a closing fence
            \n\s*```\s*(\n|$)            # Close the fence.
        )
        (                              # Optional!
            [\s\n]*                      # Any number of blank lines.
            ```\s*\n                     # Open ```
            (?P<output>{no_fence})       # Anything except a closing fence
            \n\s*```                     # Close the fence.
        )?
        """,
            # Multiline so ^ matches after a newline
            re.MULTILINE |
            # Dotall so `.` matches newlines.
            re.DOTALL |
            # Verbose to allow comments/ignore-whitespace.
            re.VERBOSE,
        )

    def get_examples(
        self, string: str, name: str = "<string>"
    ) -> List[doctest.Example]:
        tests = []
        for match in self.fence_cell_re.finditer(string):
            if re.search("doctest.*skip", match.group(0), re.IGNORECASE):
                continue

            # Do not test any docstring with our format string markers.
            # These will not run until formatted.
            if re.search("{{", match.group(0)):
                continue

            groups = match.groupdict()

            source = textwrap.dedent(groups["doctest"])
            want = groups["output"]
            if want is not None:
                want = textwrap.dedent(want)

            tests.append(
                doctest.Example(
                    lineno=string[: match.start()].count("\n") + 1,
                    source=source,
                    want=want,
                )
            )
        return tests


def _print_if_not_none(obj):
    """Print like a notebook: Show the repr if the object is not None.

    `_patch_compile` Uses this on the final expression in each cell.

    This way the outputs feel like notebooks.

    Args:
      obj: the object to print.
    """
    if obj is not None:
        print(repr(obj))


def _patch_compile(
    source, filename, mode, flags=0, dont_inherit=False, optimize=-1
):
    """Patch `doctest.compile` to make doctest to behave like a notebook.

    Default settings for doctest are configured to run like a repl: one statement
    at a time. The doctest source uses `compile(..., mode="single")`

    So to let doctest act like a notebook:

    1. We need `mode="exec"` (easy)
    2. We need the last expression to be printed (harder).

    To print the last expression, just wrap the last expression in
    `_print_if_not_none(expr)`. To detect the last expression use `AST`.
    If the last node is an expression modify the ast to call
    `_print_if_not_none` on it, convert the ast back to source and compile that.

    https://docs.python.org/3/library/functions.html#compile

    Args:
      source: Can either be a normal string, a byte string, or an AST object.
      filename: Argument should give the file from which the code was read; pass
        some recognizable value if it wasn’t read from a file ('<string>' is
        commonly used).
      mode: [Ignored] always use exec.
      flags: Compiler options.
      dont_inherit: Compiler options.
      optimize: Compiler options.

    Returns:
      The resulting code object.
    """
    # doctest passes some dummy string as the file name, AFAICT
    # but tf.function freaks-out if this doesn't look like a
    # python file name.
    del filename
    # Doctest always passes "single" here, you need exec for multiple lines.
    del mode

    source_ast = ast.parse(source)

    final = source_ast.body[-1]
    if isinstance(final, ast.Expr):
        # Wrap the final expression as `_print_if_not_none(expr)`
        print_it = ast.Expr(
            lineno=-1,
            col_offset=-1,
            value=ast.Call(
                func=ast.Name(
                    id="_print_if_not_none",
                    ctx=ast.Load(),
                    lineno=-1,
                    col_offset=-1,
                ),
                lineno=-1,
                col_offset=-1,
                args=[final],  # wrap the final Expression
                keywords=[],
            ),
        )
        source_ast.body[-1] = print_it

        # It's not clear why this step is necessary. `compile` is supposed to handle
        # AST directly.
        source = astor.to_source(source_ast)

    return compile(
        source,
        filename="dummy.py",
        mode="exec",
        flags=flags,
        dont_inherit=dont_inherit,
        optimize=optimize,
    )
