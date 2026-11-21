#!/usr/bin/env python3
"""
Author: [Laxma Reddy Patlolla](https://www.github.com/laxmareddyp)
Date: 2025-12-23
Title: Smart Hugging Face to Keras-hub Model Porter

This script automatically:
1. Analyzes the complete KerasHub structure for any reference model
2. Understands file dependencies automatically
3. Generates files in the correct dependency order
4. Includes utility files (checkpoint conversion, transformers conversion)
5. Follows KerasHub structure and format strictly
6. Generates complete files without truncation
7. Uses AST parsing to extract function and class interfaces for accurate
   dependency handling

Usage Examples:
# Use Gemini (default) with output directory
python tools/model_porter.py --model_name qwen3 --reference_model mixtral \
    --api_key YOUR_GEMINI_KEY --output_dir qwen3

# Use Claude with output directory
python tools/model_porter.py --model_name qwen3 --reference_model mixtral \
    --api_key YOUR_CLAUDE_KEY --api_provider claude --output_dir qwen3

# Use OpenAI with output directory
python tools/model_porter.py --model_name qwen3 --reference_model mixtral \
    --api_key YOUR_OPENAI_KEY --api_provider openai --output_dir qwen3

# Quick reference: Always use --output_dir MODEL_NAME for organized file
# structure
"""

import argparse
import ast
import re
import sys
import time
from collections import OrderedDict
from collections import deque
from pathlib import Path

import requests


class SmartHFToKerasHubPorter:
    """Smart porter that analyzes dependencies and generates files in correct
    order.
    """

    def __init__(
        self,
        api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
        api_provider="gemini",
    ):
        """Initialize the porter.

        Args:
            api_key: API key for the selected provider.
            base_url: API base URL (only used for Gemini).
            api_provider: API provider to use (default: gemini).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.api_provider = api_provider
        self.headers = {
            "Content-Type": "application/json",
        }

        # HF Transformers GitHub base URLs
        self.hf_base_url = (
            "https://raw.githubusercontent.com/huggingface/transformers/main"
        )
        self.hf_modular_path = "src/transformers/models"

        # KerasHub local paths
        self.kh_base_path = Path(__file__).parent

        # File patterns to look for
        self.file_patterns = {
            "core": ["*.py"],
            "checkpoint_conversion": ["convert_*_checkpoints.py"],
            "transformers_utils": ["convert_*.py"],
            "tests": ["*_test.py"],
            "presets": ["*_presets.py"],
            "tokenizers": ["*_tokenizer.py"],
        }

        # Core file dependencies - defines which files depend on which other
        # files. This uses canonical filenames, ie. without model-specific
        # prefix.
        self.core_file_dependencies = {
            "__init__.py": [],
            "layer_norm.py": [],
            "attention.py": ["layer_norm.py"],
            "decoder.py": ["attention.py", "layer_norm.py"],
            "backbone.py": ["decoder.py", "layer_norm.py"],
            "causal_lm.py": ["backbone.py", "tokenizer.py"],
            "causal_lm_preprocessor.py": ["backbone.py", "tokenizer.py"],
            "tokenizer.py": ["backbone.py"],
            "presets.py": ["backbone.py"],
            "convert_model.py": ["backbone.py"],
            "convert_model_checkpoints.py": ["backbone.py", "tokenizer.py"],
            "backbone_test.py": ["backbone.py"],
            "causal_lm_test.py": [
                "causal_lm.py",
                "backbone.py",
                "causal_lm_preprocessor.py",
                "tokenizer.py",
            ],
            "causal_lm_preprocessor_test.py": [
                "backbone.py",
                "causal_lm_preprocessor.py",
                "tokenizer.py",
            ],
        }

        self.core_file_aliases = {"layer_norm.py": ["layernorm.py"]}

    def find_hf_modular_file(self, model_name):
        """Find and read the relevant modular file from transformers."""
        print(f"🔍 Searching for modular file for model: {model_name}")

        possible_files = [
            f"modular_{model_name}.py",
            f"{model_name}.py",
            f"modeling_{model_name}.py",
        ]

        for filename in possible_files:
            url = (
                f"{self.hf_base_url}/{self.hf_modular_path}/{model_name}/"
                f"{filename}"
            )
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"✅ Found modular file: {url}")
                    return response.text
            except requests.RequestException as e:
                print(f"⚠️  Failed to fetch {url}: {e}")
                continue

        # Try main modeling file
        url = (
            f"{self.hf_base_url}/{self.hf_modular_path}/{model_name}/"
            f"modeling_{model_name}.py"
        )
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Found modeling file: {url}")
                return response.text
        except requests.RequestException as e:
            print(f"⚠️  Failed to fetch {url}: {e}")

        print(f"❌ No modular or modeling file found for {model_name}")
        return None

    def analyze_keras_hub_structure(self, reference_model):
        """
        Analyze the complete KerasHub structure for a reference model.

        Returns:
            Dict[str, Dict]: A dictionary with the following keys:
                - "core_files": Mapping of core model .py filenames to their
                  file contents.
                - "checkpoint_conversion": Mapping of checkpoint conversion
                  filenames to their file contents.
                - "transformers_utils": Mapping of transformers utility
                  filenames to their file contents.
                - "imports": Mapping of core model .py filenames to a list of
                  their import statements.

        The structure provides a comprehensive overview of the reference
        model's KerasHub implementation, including all core source files,
        checkpoint conversion scripts, transformers utilities, and their
        imports.
        """
        print(f"🔍 Analyzing KerasHub structure for: {reference_model}")

        structure = {
            "core_files": {},
            "checkpoint_conversion": {},
            "transformers_utils": {},
            "imports": {},
        }

        # Always fetch reference files from GitHub for consistency and
        # portability
        print(f"🌐 Fetching {reference_model} reference files from GitHub...")
        self.fetch_reference_files_from_github(reference_model, structure)

        return structure

    def fetch_reference_files_from_github(self, reference_model, structure):
        """Fetch reference files from KerasHub GitHub repository dynamically."""
        print(f"🌐 Fetching {reference_model} reference files from GitHub...")

        # KerasHub GitHub base URL
        kh_github_base = (
            "https://raw.githubusercontent.com/keras-team/keras-hub/master"
        )

        # First, try to get the directory listing from GitHub API to discover
        # all files
        try:
            # Use GitHub API to get directory contents
            api_url = f"https://api.github.com/repos/keras-team/keras-hub/contents/keras_hub/src/models/{reference_model}"
            response = requests.get(api_url, timeout=10)

            if response.status_code == 200:
                files_info = response.json()
                print(
                    f"📁 Discovered {len(files_info)} files in "
                    f"{reference_model} directory"
                )

                # Fetch each Python file
                for file_info in files_info:
                    if file_info["name"].endswith(".py"):
                        filename = file_info["name"]
                        try:
                            url = (
                                f"{kh_github_base}/keras_hub/src/models/"
                                f"{reference_model}/{filename}"
                            )
                            file_response = requests.get(url, timeout=10)
                            if file_response.status_code == 200:
                                content = file_response.text
                                structure["core_files"][filename] = content
                                print(
                                    f"✅ Fetched core file from GitHub: "
                                    f"{filename}"
                                )

                                # Analyze imports and dependencies
                                imports = self.extract_imports(content)
                                structure["imports"][filename] = imports
                            else:
                                print(
                                    f"⚠️  Could not fetch {filename} from "
                                    f"GitHub (status: "
                                    f"{file_response.status_code})"
                                )
                        except Exception as e:
                            print(
                                f"⚠️  Failed to fetch {filename} from "
                                f"GitHub: {e}"
                            )
            else:
                print(
                    f"⚠️  Could not access GitHub API for {reference_model} "
                    f"directory (status: {response.status_code})"
                )
                # Fallback to common file patterns if API fails
                self.fetch_common_files_from_github(reference_model, structure)

        except Exception as e:
            print(f"⚠️  GitHub API request failed: {e}")
            # Fallback to common file patterns if API fails
            self.fetch_common_files_from_github(reference_model, structure)

        # Fetch checkpoint conversion file
        self.fetch_checkpoint_conversion_from_github(reference_model, structure)

        # Fetch transformers utility file
        self.fetch_transformers_utility_from_github(reference_model, structure)

    def fetch_common_files_from_github(self, reference_model, structure):
        """Fallback method to fetch files when API discovery fails - tries
        alternative API endpoints.
        """
        print(f"🔄 Using fallback method to fetch {reference_model} files...")

        kh_github_base = (
            "https://raw.githubusercontent.com/keras-team/keras-hub/master"
        )

        # Try alternative API endpoints or direct file access
        try:
            # Try with different API parameters
            api_url = f"https://api.github.com/repos/keras-team/keras-hub/contents/keras_hub/src/models/{reference_model}?ref=master"
            response = requests.get(api_url, timeout=10)

            if response.status_code == 200:
                files_info = response.json()
                print(
                    f"📁 Fallback discovered {len(files_info)} files in "
                    f"{reference_model} directory"
                )

                # Fetch each Python file
                for file_info in files_info:
                    if file_info["name"].endswith(".py"):
                        filename = file_info["name"]
                        try:
                            url = (
                                f"{kh_github_base}/keras_hub/src/models/"
                                f"{reference_model}/{filename}"
                            )
                            file_response = requests.get(url, timeout=10)
                            if file_response.status_code == 200:
                                content = file_response.text
                                structure["core_files"][filename] = content
                                print(
                                    f"✅ Fetched core file from GitHub: "
                                    f"{filename}"
                                )

                                # Analyze imports and dependencies
                                imports = self.extract_imports(content)
                                structure["imports"][filename] = imports
                            else:
                                print(
                                    f"⚠️  Could not fetch {filename} from "
                                    f"GitHub (status: "
                                    f"{file_response.status_code})"
                                )
                        except Exception as e:
                            print(
                                f"⚠️  Failed to fetch {filename} from "
                                f"GitHub: {e}"
                            )
            else:
                print(
                    f"⚠️  Fallback API also failed (status: "
                    f"{response.status_code})"
                )

        except Exception as e:
            print(f"⚠️  Fallback method failed: {e}")
            print(f"❌ Unable to fetch {reference_model} files from GitHub")

    def fetch_checkpoint_conversion_from_github(
        self, reference_model, structure
    ):
        """Fetch checkpoint conversion file from GitHub."""
        kh_github_base = (
            "https://raw.githubusercontent.com/keras-team/keras-hub/master"
        )

        try:
            checkpoint_url = (
                f"{kh_github_base}/tools/checkpoint_conversion/"
                f"convert_{reference_model}_checkpoints.py"
            )
            response = requests.get(checkpoint_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                structure["checkpoint_conversion"][
                    f"convert_{reference_model}_checkpoints.py"
                ] = content
                print(
                    f"✅ Fetched checkpoint conversion from GitHub: "
                    f"convert_{reference_model}_checkpoints.py"
                )
            else:
                print(
                    f"⚠️  Could not fetch checkpoint conversion from GitHub "
                    f"(status: {response.status_code})"
                )
        except Exception as e:
            print(f"⚠️  Failed to fetch checkpoint conversion from GitHub: {e}")

    def fetch_transformers_utility_from_github(
        self, reference_model, structure
    ):
        """Fetch transformers utility file from GitHub."""
        kh_github_base = (
            "https://raw.githubusercontent.com/keras-team/keras-hub/master"
        )

        try:
            transformers_url = (
                f"{kh_github_base}/keras_hub/src/utils/transformers/"
                f"convert_{reference_model}.py"
            )
            response = requests.get(transformers_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                structure["transformers_utils"][
                    f"convert_{reference_model}.py"
                ] = content
                print(
                    f"✅ Fetched transformers utility from GitHub: "
                    f"convert_{reference_model}.py"
                )
            else:
                print(
                    f"⚠️  Could not fetch transformers utility from GitHub "
                    f"(status: {response.status_code})"
                )
        except Exception as e:
            print(f"⚠️  Failed to fetch transformers utility from GitHub: {e}")

    def extract_imports(self, content):
        """Extract import statements from Python code."""
        imports = []

        # Simple regex-based import extraction
        import_patterns = [
            r"from\s+keras_hub\.src\.models\.\w+\.(\w+)\s+import\s+([^;\n]+)",
            r"from\s+keras_hub\.src\.models\.\w+\s+import\s+([^;\n]+)",
            r"import\s+keras_hub\.src\.models\.\w+\.(\w+)",
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)

        return imports

    def extract_interfaces(self, content):
        """Extract function and class interfaces from Python code using AST.

        Replaces all function and method bodies with 'pass' to create clean
        interfaces.
        """
        try:
            # Parse the Python code
            tree = ast.parse(content)

            class InterfaceExtractor(ast.NodeTransformer):
                """AST transformer to replace function/method bodies with
                pass.
                """

                def visit_FunctionDef(self, node):
                    """Replace function body with pass."""
                    # Keep the function signature but replace body with pass
                    node.body = [ast.Pass()]
                    return node

                def visit_AsyncFunctionDef(self, node):
                    """Replace async function body with pass."""
                    node.body = [ast.Pass()]
                    return node

                def visit_ClassDef(self, node):
                    """Process class methods but keep class body structure."""
                    # Process all methods in the class
                    for item in node.body:
                        if isinstance(
                            item, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            item.body = [ast.Pass()]
                    return node

                def visit_If(self, node):
                    """Replace if statement body with pass."""
                    node.body = [ast.Pass()]
                    if node.orelse:
                        node.orelse = [ast.Pass()]
                    return node

                def visit_For(self, node):
                    """Replace for loop body with pass."""
                    node.body = [ast.Pass()]
                    if node.orelse:
                        node.orelse = [ast.Pass()]
                    return node

                def visit_While(self, node):
                    """Replace while loop body with pass."""
                    node.body = [ast.Pass()]
                    if node.orelse:
                        node.orelse = [ast.Pass()]
                    return node

                def visit_Try(self, node):
                    """Replace try-except body with pass."""
                    node.body = [ast.Pass()]
                    for handler in node.handlers:
                        handler.body = [ast.Pass()]
                    if node.orelse:
                        node.orelse = [ast.Pass()]
                    if node.finalbody:
                        node.finalbody = [ast.Pass()]
                    return node

                def visit_With(self, node):
                    """Replace with statement body with pass."""
                    node.body = [ast.Pass()]
                    return node

                def visit_Expr(self, node):
                    """Keep expressions (like docstrings) but remove other
                    statements.
                    """
                    if isinstance(node.value, ast.Str):
                        return node  # Keep docstrings
                    return ast.Pass()

                def visit_Assign(self, node):
                    """Keep assignments (like class variables)."""
                    return node

                def visit_AnnAssign(self, node):
                    """Keep annotated assignments."""
                    return node

                def visit_Import(self, node):
                    """Keep import statements."""
                    return None

                def visit_ImportFrom(self, node):
                    """Keep from-import statements."""
                    return None

            # Transform the AST
            transformer = InterfaceExtractor()
            transformed_tree = transformer.visit(tree)

            # Convert back to Python code
            interface_code = ast.unparse(transformed_tree)

            return interface_code

        except Exception as e:
            print(f"⚠️  Failed to extract interfaces using AST: {e}")
            # Fallback: return original content if AST parsing fails
            return content

    def topological_sort(self, graph):
        # Step 1: Calculate in-degree of each node
        in_degree = {node: 0 for node in graph}
        for deps in graph.values():
            for dep in deps:
                if dep not in in_degree:
                    in_degree[dep] = 0
                in_degree[dep] += 1

        # Step 2: Initialize queue with nodes of in-degree 0
        queue = deque(
            [node for node, degree in in_degree.items() if degree == 0]
        )

        sorted_list = []
        while queue:
            node = queue.popleft()
            sorted_list.append(node)

            # Reduce in-degree of neighbors
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Step 3: Check for cycle (if not all nodes processed)
        if len(sorted_list) != len(in_degree):
            raise ValueError("Graph has a cycle, topological sort not possible")

        return sorted_list

    def determine_target_files(
        self,
        target_model,
        reference_model,
        structure,
    ):
        """Determine which files need to be generated for the target model."""
        print(f"🔍 Determining target files for: {reference_model}")

        # This will map canonical filenames (without model-specific prefix) to:
        # {
        #   "target_path": path for new model's file,
        #   "reference_name": filename for the reference model,
        #   "dependencies": [list of canonical filenames of dependencies]
        # }

        # We'll use canonical filenames (e.g., "backbone.py") and map to
        # target/reference filenames
        canonical_to_target = {}
        canonical_to_reference = {}
        dependencies_map = {}

        # 1. Iterate the list of core_file_dependencies
        for canonical_name in self.core_file_dependencies:
            # 2. Find the corresponding target model filename
            # e.g., "backbone.py"  -> "qwen3_backbone.py" for target
            # Special handling for convert_ files (e.g., convert_model.py,
            # convert_model_checkpoints.py)
            # For checkpoint conversion files, use the correct naming pattern
            if canonical_name == "convert_model_checkpoints.py":
                target_file = f"convert_{target_model}_checkpoints.py"
            elif canonical_name == "convert_model.py":
                target_file = f"convert_{target_model}.py"
            elif canonical_name == "__init__.py":
                target_file = "__init__.py"
            else:
                target_file = f"{target_model}_{canonical_name}"

            # Find the reference model filename, e.g. "backbone.py" ->
            # "mixtral_backbone.py"
            for fn in [canonical_name] + self.core_file_aliases.get(
                canonical_name, []
            ):
                if fn == "convert_model_checkpoints.py":
                    reference_file = f"convert_{reference_model}_checkpoints.py"
                elif fn == "convert_model.py":
                    reference_file = f"convert_{reference_model}.py"
                elif fn == "__init__.py":
                    reference_file = "__init__.py"
                else:
                    reference_file = f"{reference_model}_{fn}"
                if any(
                    reference_file in structure[key]
                    for key in [
                        "core_files",
                        "checkpoint_conversion",
                        "transformers_utils",
                    ]
                ):
                    break
            else:
                continue

            # 3. Add the canonical filename to the result
            canonical_to_target[canonical_name] = target_file
            canonical_to_reference[canonical_name] = reference_file

            # 4. Build the dependencies (as canonical names)
            dependencies_map[canonical_name] = self.core_file_dependencies[
                canonical_name
            ]

        # 5. Reorder the result by dependencies (topological sort)
        result = OrderedDict()
        for canonical_name in self.topological_sort(dependencies_map)[::-1]:
            result[canonical_name] = {
                "target_path": canonical_to_target[canonical_name],
                "reference_name": canonical_to_reference[canonical_name],
                "dependencies": dependencies_map[canonical_name],
            }

        return result

    def generate_file_prompt(
        self,
        target_file,
        reference_file,
        reference_content,
        pytorch_code,
        target_model,
        reference_model,
        dependencies,
        structure,
    ):
        """
        Generate a comprehensive prompt for file generation.

        Args:
            target_file (str): The filename to generate for the target model
                (e.g., "qwen3_backbone.py").
            reference_file (str): The corresponding reference model filename
                (e.g., "mixtral_backbone.py").
            reference_content (str): The full content of the reference file.
            pytorch_code (str): The PyTorch (HF) code for the target model.
            target_model (str): The name of the target model (e.g., "qwen3").
            reference_model (str): The name of the reference model
                (e.g., "mixtral").
            dependencies (Dict[str, str]): Dict mapping the model's dependency
                filenames to the respective class interfaces.
            structure (Dict[str, Dict]): The analyzed KerasHub structure for
                the reference model.

        Returns:
            str: The generated prompt string to be sent to the LLM.
        """

        # Enhanced prompt based on file type
        if "backbone" in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain the main {target_model.title()}Backbone
class that inherits from Backbone.
It should include:
- Complete __init__ method with all parameters
- Complete call method
- Complete get_config method
- All necessary imports and dependencies
- Proper KerasHub export decorator
- Lines must not exceed 80 characters
"""

        elif "attention" in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain the complete attention implementation:
- {target_model.title()}Attention class with sliding window attention
- Sink token handling
- Rotary embeddings integration
- Complete call method with all parameters
- Lines must not exceed 80 characters
"""
        elif "decoder" in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain the complete decoder implementation:
- {target_model.title()}TransformerDecoder class
- All layer methods and forward passes
- MoE integration
- Complete implementation, no placeholders
- Lines must not exceed 80 characters
"""
        elif "checkpoints" in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain the complete checkpoint conversion script:
- Complete convert_backbone_config function
- Complete convert_weights function with MoE weight handling
- Complete convert_tokenizer function
- All necessary imports and dependencies
- No placeholder comments - implement everything completely
- Handle MoE expert weights properly
- Include proper error handling and validation
- Match the structure of the reference {reference_model} checkpoint
conversion script
- IMPORTANT: Use {target_model.title()} model names in all keras_hub.model calls
- IMPORTANT: Use {target_model.title()}CausalLMPreprocessor,
{target_model.title()}CausalLM, etc.
- IMPORTANT: Do NOT use the reference model names
- Lines must not exceed 80 characters
"""
        elif "convert" in target_file and "transformers" not in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain the complete transformers utility:
- Complete convert_backbone_config function
- Complete convert_weights function
- Complete convert_tokenizer function
- All necessary imports and dependencies
- No placeholder comments - implement everything completely
- Match the structure of the reference {reference_model} transformers utility
- IMPORTANT: Import from the CORRECT model: from
keras_hub.src.models.{target_model}.{target_model}_backbone import
{target_model.title()}Backbone
- IMPORTANT: Set backbone_cls = {target_model.title()}Backbone
- Lines must not exceed 80 characters
"""
        elif "__init__" in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain ONLY imports and exports - NO class
implementations:
- Import main classes using relative imports: from .{target_model}_backbone
import {target_model.title()}Backbone
- Import presets: from .{target_model}_presets import backbone_presets
- Import preset_utils: from keras_hub.src.utils.preset_utils import
register_presets
- Register presets: register_presets(backbone_presets,
{target_model.title()}Backbone)
- Follow EXACTLY the structure of the reference mixtral __init__.py
- The file should be ONLY 4-5 lines of imports and registration
- NO class definitions, NO method implementations, NO other code
- This is just a module initialization file
- Lines must not exceed 80 characters
"""
        elif "layer_norm" in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain the complete layer normalization
implementation:
- {target_model.title()}LayerNormalization class with RMS normalization
- Use ONLY keras and keras_hub imports
- Complete __init__, build, call, and get_config methods
- Proper epsilon handling and scale parameter
- Follow KerasHub conventions exactly
- Lines must not exceed 80 characters
"""
        elif "causal_lm_preprocessor" in target_file:
            file_specific_instructions = f"""
CRITICAL: This file MUST contain the complete causal LM preprocessor:
- Import from {target_model}_backbone and {target_model}_tokenizer
- Set backbone_cls = {target_model.title()}Backbone
- Set tokenizer_cls = {target_model.title()}Tokenizer
- Complete Examples section in docstring with full code examples
- Include backbone_cls and tokenizer_cls class attributes
- Include __init__ method
- NO TRUNCATION - the file must be complete from start to finish
- Generate the ENTIRE file including all methods and docstrings
- Lines must not exceed 80 characters
"""
        else:
            file_specific_instructions = """
CRITICAL: Generate the COMPLETE file with all classes, methods, and
functionality.
No placeholder comments, no incomplete implementations.
"""

        # Extract dependency interfaces
        dependency_interfaces = ""
        if dependencies:
            dependency_interfaces = "\n\nDependency Interfaces:\n"
            dependency_interfaces += (
                "The following are the interfaces of files this file depends "
                "on:\n"
            )
            dependency_interfaces += (
                "Use these interfaces to ensure proper integration:\n\n"
            )
            dependency_interfaces += "\n\n".join(
                [
                    f"--- {dep} Interface ---\n{dep_int}"
                    for dep, dep_int in dependencies.items()
                ]
            )

        prompt = f"""Generate a COMPLETE KerasHub {target_file} file for the
        {target_model} model.

Reference {reference_model} file: {reference_file}

Reference file content:
{reference_content}

PyTorch {target_model} code to adapt:
{pytorch_code}

Dependencies this file needs:
{", ".join(dependencies) if dependencies else "None"}

{file_specific_instructions}

Requirements:
1. Generate COMPLETE code - do not truncate or leave incomplete
2. Follow KerasHub conventions and import patterns EXACTLY
3. Adapt PyTorch classes to Keras layers
4. Include all necessary methods and functionality
5. Use proper KerasHub naming conventions
6. Ensure the file is self-contained and importable
7. Match the structure and style of the reference file
8. Handle all dependencies properly
9. NO PLACEHOLDER COMMENTS - implement everything completely
10. Include all imports and exports needed
11. Lines must not exceed 80 characters

{dependency_interfaces}

Generate the complete {target_file} file with no truncation:"""

        return prompt

    def call_gemini_api(self, prompt):
        """Call Gemini API with maximum token limit to avoid truncation."""
        print("🤖 Calling Gemini API...")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 131072,  # Increased to 131K tokens
            },
        }

        try:
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(
                url, headers=self.headers, json=payload, timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        generated_text = content["parts"][0]["text"]
                        print("✅ Successfully generated code from Gemini API")
                        return generated_text
                    else:
                        print("❌ No content parts in Gemini response")
                        return None
                else:
                    print("❌ No candidates in Gemini response")
                    return None
            else:
                print(
                    f"❌ Gemini API request failed with status "
                    f"{response.status_code}"
                )
                print(f"Response: {response.text}")
                return None

        except requests.RequestException as e:
            print(f"❌ Failed to call Gemini API: {e}")
            return None

    def call_claude_api(self, prompt):
        """Call Claude API with maximum token limit to avoid truncation."""
        print("🤖 Calling Claude API...")

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 131072,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                if "content" in result and len(result["content"]) > 0:
                    generated_text = result["content"][0]["text"]
                    print("✅ Successfully generated code from Claude API")
                    return generated_text
                else:
                    print("❌ No content in Claude response")
                    return None
            else:
                print(
                    f"❌ Claude API request failed with status "
                    f"{response.status_code}"
                )
                print(f"Response: {response.text}")
                return None

        except requests.RequestException as e:
            print(f"❌ Failed to call Claude API: {e}")
            return None

    def call_openai_api(self, prompt):
        """Call OpenAI API with maximum token limit to avoid truncation."""
        print("🤖 Calling OpenAI API...")

        payload = {
            "model": "gpt-4o-mini",
            "max_tokens": 131072,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0]["message"]["content"]
                    print("✅ Successfully generated code from OpenAI API")
                    return generated_text
                else:
                    print("❌ No choices in OpenAI response")
                    return None
            else:
                print(
                    f"❌ OpenAI API request failed with status "
                    f"{response.status_code}"
                )
                print(f"Response: {response.text}")
                return None

        except requests.RequestException as e:
            print(f"❌ Failed to call OpenAI API: {e}")
            return None

    def call_api(self, prompt):
        """Route to the appropriate API based on provider."""
        if self.api_provider == "claude":
            return self.call_claude_api(prompt)
        elif self.api_provider == "openai":
            return self.call_openai_api(prompt)
        else:
            return self.call_gemini_api(prompt)

    def extract_generated_code(self, generated_text, target_file):
        """Extract generated code from Gemini response."""
        print("🔧 Extracting generated code...")

        # Try to find Python code blocks
        # Greedy matching, since docstrings might contain code blocks
        code_patterns = [
            r"```python\s*(.*)\s*```",
            r"```\s*(.*)\s*```",
            # More flexible patterns to catch edge cases
            r"```python\s*(.*?)(?=\s*```|$)",
            r"```\s*(.*?)(?=\s*```|$)",
        ]

        for pattern in code_patterns:
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if (
                    "import keras" in code
                    or "\nfrom keras_hub." in code
                    or "class " in code
                ):
                    print("✅ Successfully extracted generated code")
                    return code

        # If no code blocks found, try to extract based on import statements
        if "import keras" in generated_text:
            # Find the start of the code (first import)
            start_idx = generated_text.find("import keras")
            code = generated_text[start_idx:]

            # Don't truncate - return the full content from first import onwards
            print("✅ Extracted code based on import statements (full content)")
            return code

        # Clean up the full response to remove markdown artifacts
        cleaned_code = generated_text

        # Remove markdown code block syntax if present
        cleaned_code = re.sub(
            r"^```python\s*", "", cleaned_code, flags=re.MULTILINE
        )
        cleaned_code = re.sub(r"^```\s*", "", cleaned_code, flags=re.MULTILINE)
        cleaned_code = re.sub(
            r"\s*```\s*$", "", cleaned_code, flags=re.MULTILINE
        )

        # Validate that we have meaningful content
        if len(cleaned_code.strip()) > 100 and (
            "class " in cleaned_code or "def " in cleaned_code
        ):
            print("✅ Returning cleaned full response with validation")
            return cleaned_code
        else:
            print(
                "⚠️  Could not extract code blocks, returning cleaned full "
                "response"
            )
            return cleaned_code

    def save_generated_file(
        self,
        code,
        target_file,
        target_model,
        output_dir=None,
    ):
        """Save generated code to the specified output directory."""
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / target_file

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"💾 Saved generated code to: {output_path}")
        return str(output_path)

    def generate_complete_model(
        self, target_model, reference_model, output_dir=None
    ):
        """Generate the complete model with all dependencies."""
        print(
            f"🚀 Starting generation of complete {target_model} model using "
            f"{reference_model} as reference"
        )
        print("=" * 80)

        # Step 1: Get PyTorch code
        pytorch_code = self.find_hf_modular_file(target_model)
        if not pytorch_code:
            print("❌ Failed to find HF modular file")
            return {}

        # Step 2: Analyze KerasHub structure
        structure = self.analyze_keras_hub_structure(reference_model)
        if not structure["core_files"]:
            print("❌ Failed to find KerasHub reference files")
            return {}

        # Step 3: Determine target files
        target_files = self.determine_target_files(
            target_model, reference_model, structure
        )
        # Step 4: Generate files in dependency order
        results = {}

        for canonical_name, file_info in target_files.items():
            target_file = file_info["target_path"]
            reference_file = file_info["reference_name"]
            file_deps = file_info["dependencies"]

            print(f"\n🔄 Generating {target_file}...")
            print("-" * 50)

            try:
                # Get reference file content from structure
                reference_content = (
                    structure["core_files"].get(reference_file)
                    or structure["checkpoint_conversion"].get(reference_file)
                    or structure["transformers_utils"].get(reference_file)
                )

                # Build the dependencies dict for this file
                dependent_interfaces = {}
                for dep in file_deps:
                    # Map canonical dependency name to target model filename
                    dep_target_file = (
                        target_files[dep]["target_path"]
                        if dep in target_files
                        else dep
                    )
                    # Try to find the generated file for this dependency
                    dep_path = Path(output_dir) / dep_target_file
                    try:
                        dep_content = dep_path.read_text(encoding="utf-8")
                        # Extract the interface from the dependency file
                        dependent_interfaces[dep_target_file] = (
                            self.extract_interfaces(dep_content)
                        )
                    except Exception as e:
                        print(
                            f"⚠️  Failed to extract interface from "
                            f"{dep_target_file}: {e}"
                        )

                # Generate prompt
                prompt = self.generate_file_prompt(
                    target_file,
                    reference_file,
                    reference_content,
                    pytorch_code,
                    target_model,
                    reference_model,
                    dependent_interfaces,
                    structure,
                )

                # Call API
                generated_text = self.call_api(prompt)
                if not generated_text:
                    print(f"❌ Failed to generate {target_file}")
                    results[target_file] = False
                    continue

                # Extract code
                extracted_code = self.extract_generated_code(
                    generated_text, target_file
                )

                # Save file
                self.save_generated_file(
                    extracted_code, target_file, target_model, output_dir
                )

                results[target_file] = True
                print(f"✅ Successfully generated {target_file}")

                # Add delay between API calls to avoid rate limiting
                time.sleep(2)

            except Exception as e:
                print(f"❌ Error generating {target_file}: {e}")
                results[target_file] = False

        return results

    def validate_model_name(self, model_name):
        """Validate model name."""
        if not re.match(r"^[a-zA-Z0-9_]+$", model_name):
            return False
        if len(model_name) < 2 or len(model_name) > 50:
            return False
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smart HF to KerasHub Model Porter - Generates complete "
        "models "
        "with dependencies using Gemini, Claude, or OpenAI APIs"
    )

    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the HF model to port (e.g., 'deepseek_v2', 'qwen3', "
        "'gpt_oss')",
    )

    parser.add_argument(
        "--reference_model",
        required=True,
        help="Name of the reference model in KerasHub (e.g., 'mixtral')",
    )

    parser.add_argument(
        "--api_key", required=True, help="API key for the selected provider"
    )

    parser.add_argument(
        "--api_provider",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider to use (default: gemini)",
    )

    parser.add_argument(
        "--output_dir",
        help="Output directory for generated files (defaults to current "
        "directory)",
    )

    parser.add_argument(
        "--base_url",
        default="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
        help="API base URL (only used for Gemini)",
    )

    args = parser.parse_args()

    # Validate inputs
    porter = SmartHFToKerasHubPorter(
        args.api_key, args.base_url, args.api_provider
    )

    if not porter.validate_model_name(args.model_name):
        print(f"❌ Invalid model name: {args.model_name}")
        sys.exit(1)

    if not porter.validate_model_name(args.reference_model):
        print(f"❌ Invalid reference model name: {args.reference_model}")
        sys.exit(1)

    # Generate complete model
    results = porter.generate_complete_model(
        args.model_name, args.reference_model, args.output_dir
    )

    # Summary
    print("\n" + "=" * 80)
    print("📊 Generation Summary")
    print("=" * 80)

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    for target_file, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {target_file}")

    print(
        f"\n🎯 Success Rate: {successful}/{total} "
        f"({successful / total * 100:.1f}%)"
    )

    if successful == total:
        print("\n🎉 Complete model generated successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - successful} files failed to generate")
        sys.exit(1)


if __name__ == "__main__":
    main()
