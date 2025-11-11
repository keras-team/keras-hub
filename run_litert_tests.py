#!/usr/bin/env python3
"""
Script to run all LiteRT export tests for Keras Hub models and update
coverage documentation.

This script:
1. Discovers all test files containing test_litert_export methods
2. Runs each test and collects pass/fail results
3. Updates the keras_hub_litert_coverage.md file with current status
4. Identifies models without tests
"""

import os
import subprocess
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

# Test files with test_litert_export methods (from grep search results)
INDIVIDUAL_TEST_FILES = [
    "keras_hub/src/models/gpt2/gpt2_causal_lm_test.py",
    "keras_hub/src/models/mit/mit_image_classifier_test.py",
    "keras_hub/src/models/vgg/vgg_image_classifier_test.py",
    "keras_hub/src/models/mistral/mistral_causal_lm_test.py",
    "keras_hub/src/models/hgnetv2/hgnetv2_image_classifier_test.py",
    "keras_hub/src/models/xception/xception_image_classifier_test.py",
    "keras_hub/src/models/roberta/roberta_text_classifier_test.py",
    "keras_hub/src/models/deberta_v3/deberta_v3_text_classifier_test.py",
    "keras_hub/src/models/vit/vit_image_classifier_test.py",
    "keras_hub/src/models/retinanet/retinanet_object_detector_test.py",
    "keras_hub/src/models/deit/deit_image_classifier_test.py",
    "keras_hub/src/models/xlm_roberta/xlm_roberta_text_classifier_test.py",
    "keras_hub/src/models/d_fine/d_fine_object_detector_test.py",
    "keras_hub/src/models/qwen3/qwen3_causal_lm_test.py",
    "keras_hub/src/models/resnet/resnet_image_classifier_test.py",
    "keras_hub/src/models/f_net/f_net_text_classifier_test.py",
    "keras_hub/src/models/efficientnet/efficientnet_image_classifier_test.py",
    "keras_hub/src/models/gemma3/gemma3_causal_lm_test.py",
    "keras_hub/src/models/phi3/phi3_causal_lm_test.py",
    "keras_hub/src/models/roformer_v2/roformer_v2_text_classifier_test.py",
    "keras_hub/src/models/mobilenet/mobilenet_image_classifier_test.py",
    "keras_hub/src/models/gemma/gemma_causal_lm_test.py",
    "keras_hub/src/models/albert/albert_text_classifier_test.py",
    "keras_hub/src/models/llama3/llama3_causal_lm_test.py",
    "keras_hub/src/models/distil_bert/distil_bert_text_classifier_test.py",
    "keras_hub/src/models/cspnet/cspnet_image_classifier_test.py",
    "keras_hub/src/models/sam/sam_image_segmenter_test.py",
    "keras_hub/src/models/bert/bert_text_classifier_test.py",
    "keras_hub/src/models/bloom/bloom_causal_lm_test.py",
    "keras_hub/src/models/bart/bart_seq_2_seq_lm_test.py",
    "keras_hub/src/models/falcon/falcon_causal_lm_test.py",
    "keras_hub/src/models/opt/opt_causal_lm_test.py",
    "keras_hub/src/models/gpt_neo_x/gpt_neo_x_causal_lm_test.py",
    "keras_hub/src/models/llama/llama_causal_lm_test.py",
    "keras_hub/src/models/mixtral/mixtral_causal_lm_test.py",
    "keras_hub/src/models/qwen/qwen_causal_lm_test.py",
    "keras_hub/src/models/qwen_moe/qwen_moe_causal_lm_test.py",
    "keras_hub/src/models/qwen3_moe/qwen3_moe_causal_lm_test.py",
    "keras_hub/src/models/smollm3/smollm3_causal_lm_test.py",
    "keras_hub/src/models/esm/esm_classifier_test.py",
    "keras_hub/src/models/basnet/basnet_test.py",
    "keras_hub/src/models/depth_anything/depth_anything_depth_estimator_test.py",
    "keras_hub/src/models/t5gemma/t5gemma_seq_2_seq_lm_test.py",
    "keras_hub/src/models/segformer/segformer_image_segmenter_tests.py",
    "keras_hub/src/models/pali_gemma/pali_gemma_causal_lm_test.py",
    "keras_hub/src/models/stable_diffusion_3/stable_diffusion_3_text_to_image_test.py",
    "keras_hub/src/models/moonshine/moonshine_audio_to_text_test.py",
    "keras_hub/src/models/parseq/parseq_causal_lm_test.py",
]

# Parametrized test file
PARAMETRIZED_TEST_FILE = "keras_hub/src/export/litert_models_test.py"

# Markdown file to update
MARKDOWN_FILE = "keras_hub_litert_coverage.md"


def run_test(
    test_file: str, test_method: str = None, backend: str = "tensorflow"
) -> Tuple[bool, str]:
    """
    Run a specific test and return (passed, output).

    Args:
        test_file: Path to the test file
        test_method: Specific test method to run (optional)
        backend: Backend to use ('tensorflow' or 'jax')

    Returns:
        Tuple of (passed: bool, output: str)
    """
    # Set environment variable for backend
    env = os.environ.copy()
    if backend == "jax":
        env["KERAS_BACKEND"] = "jax"
    elif backend == "tensorflow":
        env["KERAS_BACKEND"] = "tensorflow"

    cmd = [
        "python3",
        "-m",
        "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--run_large",
    ]

    if test_method:
        cmd.extend(["-k", test_method])

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 5 minutes"
    except Exception as e:
        return False, f"Error running test: {str(e)}"


def extract_model_name_from_test_file(test_file: str) -> str:
    """Extract model name from test file path."""
    # e.g., "keras_hub/src/models/gpt2/gpt2_causal_lm_test.py" ->
    # "gpt2_causal_lm"
    parts = Path(test_file).parts
    if "models" in parts:
        model_idx = parts.index("models")
        if model_idx + 1 < len(parts):
            model_name = parts[model_idx + 1]
            return model_name
    return Path(test_file).stem.replace("_test", "")


def categorize_model(model_name: str) -> str:
    """Categorize model type based on name."""
    if (
        "causal_lm" in model_name
        or "gpt2" in model_name
        or "mistral" in model_name
        or "gemma" in model_name
        or "llama" in model_name
        or "phi3" in model_name
        or "qwen" in model_name
    ):
        return "CausalLM"
    elif (
        "text_classifier" in model_name
        or "bert" in model_name
        or "roberta" in model_name
        or "albert" in model_name
        or "deberta" in model_name
        or "f_net" in model_name
        or "roformer" in model_name
        or "xlm_roberta" in model_name
        or "distil_bert" in model_name
    ):
        return "TextClassifier"
    elif (
        "image_classifier" in model_name
        or "resnet" in model_name
        or "efficientnet" in model_name
        or "densenet" in model_name
        or "mobilenet" in model_name
        or "vgg" in model_name
        or "vit" in model_name
        or "deit" in model_name
        or "xception" in model_name
        or "mit" in model_name
        or "hgnetv2" in model_name
        or "cspnet" in model_name
    ):
        return "ImageClassifier"
    elif (
        "object_detector" in model_name
        or "retinanet" in model_name
        or "d_fine" in model_name
    ):
        return "ObjectDetector"
    elif "image_segmenter" in model_name or "sam" in model_name:
        return "ImageSegmenter"
    else:
        return "Unknown"


def parse_parametrized_test_output(output: str) -> Dict[str, bool]:
    """
    Parse pytest output from parametrized tests to extract individual
    test results.

    Args:
        output: Raw pytest output string

    Returns:
        Dict mapping test IDs to pass/fail status
    """
    results = {}
    lines = output.split("\n")

    for line in lines:
        line = line.strip()
        # Look for lines like:
        # "test_causal_lm_litert_export[gpt2_base_en-gpt2_base_en] PASSED"
        # or: "test_image_classifier_litert_export[resnet_50-resnet_50_imagenet]
        # FAILED"
        if "test_" in line and (
            "PASSED" in line or "FAILED" in line or "ERROR" in line
        ):
            # Extract test name and status
            parts = line.split()
            if len(parts) >= 2:
                test_full_name = parts[0]
                status = parts[-1]

                # Extract the parametrized part: [test_name-preset]
                if "[" in test_full_name and "]" in test_full_name:
                    param_part = test_full_name.split("[")[1].split("]")[0]
                    # param_part looks like "gpt2_base_en-gpt2_base_en"
                    # We want the first part as the model identifier
                    model_id = param_part.split("-")[0]

                    passed = status == "PASSED"
                    results[model_id] = passed

    return results


def run_all_tests(backend: str = "tensorflow") -> Dict[str, Dict]:
    """
    Run all LiteRT export tests and collect results.

    Args:
        backend: Backend to use ('tensorflow' or 'jax')

    Returns:
        Dict mapping model names to test results
    """
    results = {}

    print(f"Running individual model tests with {backend} backend...")
    for test_file in INDIVIDUAL_TEST_FILES:
        if not Path(test_file).exists():
            print(f"Warning: Test file {test_file} not found, skipping")
            continue

        model_name = extract_model_name_from_test_file(test_file)
        print(f"Running test for {model_name}...")

        # Handle special case for gemma3 which has two test methods
        if "gemma3" in model_name:
            # Run both test methods
            passed1, output1 = run_test(
                test_file, "test_litert_export", backend
            )
            passed2, output2 = run_test(
                test_file, "test_litert_export_multimodal", backend
            )
            passed = passed1 and passed2
            output = output1 + "\n" + output2
        else:
            passed, output = run_test(test_file, "test_litert_export", backend)

        results[model_name] = {
            "passed": passed,
            "output": output,
            "category": categorize_model(model_name),
            "test_file": test_file,
        }

        status = "PASSED" if passed else "FAILED"
        print(f"  {model_name}: {status}")

    print(f"\nRunning parametrized tests with {backend} backend...")
    if Path(PARAMETRIZED_TEST_FILE).exists():
        passed, output = run_test(PARAMETRIZED_TEST_FILE, backend=backend)
        print(f"Parametrized tests overall: {'PASSED' if passed else 'FAILED'}")

        # Parse individual test results from parametrized output
        param_results = parse_parametrized_test_output(output)

        # Add individual parametrized test results to results dict
        for model_id, test_passed in param_results.items():
            # Map model_id back to proper model name if needed
            model_name = model_id.replace(
                "_", ""
            )  # e.g., "gpt2_base_en" -> "gpt2baseen"
            # Try to find a better mapping
            if "gpt2" in model_id:
                model_name = "gpt2"
            elif "llama3" in model_id:
                model_name = "llama3"
            elif "gemma3" in model_id:
                model_name = "gemma3"
            elif "resnet" in model_id:
                model_name = "resnet"
            elif "efficientnet" in model_id:
                model_name = "efficientnet"
            elif "densenet" in model_id:
                model_name = "densenet"
            elif "mobilenet" in model_id:
                model_name = "mobilenet"
            elif "dfine" in model_id:
                model_name = "d_fine"
            elif "retinanet" in model_id:
                model_name = "retinanet"
            elif "deeplab" in model_id:
                model_name = "deeplab_v3_plus"

            results[model_name] = {
                "passed": test_passed,
                "output": f"Parametrized test result for {model_id}",
                "category": categorize_model(model_name),
                "test_file": PARAMETRIZED_TEST_FILE,
            }

            status = "PASSED" if test_passed else "FAILED"
            print(f"  {model_name} ({model_id}): {status}")

        # Store overall parametrized test result
        results["parametrized_tests"] = {
            "passed": passed,
            "output": output,
            "category": "Parametrized",
            "test_file": PARAMETRIZED_TEST_FILE,
        }
    else:
        print(
            f"Warning: Parametrized test file {PARAMETRIZED_TEST_FILE} "
            "not found"
        )

    return results


def find_models_without_tests() -> List[str]:
    """Find models that exist but don't have tests."""
    models_dir = Path("keras_hub/src/models")
    if not models_dir.exists():
        return []

    tested_models = set()
    for test_file in INDIVIDUAL_TEST_FILES:
        model_name = extract_model_name_from_test_file(test_file)
        tested_models.add(model_name)

    all_models = set()
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("__"):
            all_models.add(model_dir.name)

    return sorted(list(all_models - tested_models))


def update_markdown(
    results: Dict[str, Dict],
    models_without_tests: List[str],
    markdown_file: str = MARKDOWN_FILE,
):
    """Update the markdown file with test results."""

    # Count by category
    categories = {}
    for model_name, result in results.items():
        if model_name == "parametrized_tests":
            continue
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if result["passed"]:
            categories[cat]["passed"] += 1

    total_models = sum(cat["total"] for cat in categories.values())
    total_passed = sum(cat["passed"] for cat in categories.values())

    # Generate markdown content
    backend_name = (
        markdown_file.replace("keras_hub_litert_coverage_", "")
        .replace(".md", "")
        .upper()
    )
    content = (
        f"# Keras-Hub LiteRT Export Test Coverage ({backend_name} Backend)\n"
        "# Comprehensive list of all supported models and their "
        "LiteRT export test status\n"
        "\n"
        "## Summary:\n"
        f"- **Total Models**: {total_models}\n"
        f"- **Passed**: {total_passed}\n"
        f"- **Failed**: {total_models - total_passed}\n"
        f"- **Models without tests**: {len(models_without_tests)}\n"
        "\n"
    )

    # Add category summaries
    for cat, counts in categories.items():
        content += (
            f"## {cat} Models ({counts['passed']}/{counts['total']} passed):\n"
        )

        # Group models by status
        passed_models = []
        failed_models = []

        for model_name, result in results.items():
            if result["category"] == cat:
                if result["passed"]:
                    passed_models.append(model_name)
                else:
                    failed_models.append(model_name)

        if passed_models:
            content += "### Passed:\n"
            for model in sorted(passed_models):
                content += f"- {model} ✓\n"

        if failed_models:
            content += "### Failed:\n"
            for model in sorted(failed_models):
                content += f"- {model} ✗\n"

        content += "\n"

    # Add models without tests
    if models_without_tests:
        content += "## Models without tests:\n"
        for model in models_without_tests:
            content += f"- {model}\n"
        content += "\n"

    # Add failure details
    failed_details = []
    for model_name, result in results.items():
        if not result["passed"] and model_name != "parametrized_tests":
            failed_details.append(
                f"### {model_name}:\n```\n{result['output'][-500:]}\n```\n"
            )

    if failed_details:
        content += "## Failure Details:\n"
        content += "\n".join(failed_details)

    # Write to file
    with open(markdown_file, "w") as f:
        f.write(content)

    print(f"Updated {markdown_file}")


def main(backend: str = "tensorflow"):
    """Main function."""
    print(
        f"Starting LiteRT export test coverage analysis with "
        f"{backend} backend..."
    )

    # Update markdown filename based on backend
    markdown_file = f"keras_hub_litert_coverage_{backend}.md"

    # Run all tests
    results = run_all_tests(backend)

    # Find models without tests
    models_without_tests = find_models_without_tests()

    # Update markdown
    update_markdown(results, models_without_tests, markdown_file)

    # Print summary
    total_tests = len(
        [r for r in results.values() if r["category"] != "Parametrized"]
    )
    passed_tests = len(
        [
            r
            for r in results.values()
            if r["passed"] and r["category"] != "Parametrized"
        ]
    )

    print("\n=== SUMMARY ===")
    print(f"Total individual model tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Models without tests: {len(models_without_tests)}")

    if models_without_tests:
        print("\nModels without tests:")
        for model in models_without_tests:
            print(f"  - {model}")

    print(f"\nResults written to {markdown_file}")


if __name__ == "__main__":
    import sys

    # Check for backend argument
    backend = "tensorflow"
    if len(sys.argv) > 1:
        if sys.argv[1] in ["tensorflow", "jax"]:
            backend = sys.argv[1]
        else:
            print("Usage: python run_litert_tests.py [tensorflow|jax]")
            sys.exit(1)

    main(backend)
