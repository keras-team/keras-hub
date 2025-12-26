# Contribution guide

KerasHub is an actively growing project and community! We would love for you
to get involved. Below are instructions for how to plug into KerasHub
development.

## Background reading

Before contributing code, please review our [Style Guide](STYLE_GUIDE.md) and
[API Design Guide](API_DESIGN_GUIDE.md).

Our [Roadmap](https://github.com/keras-team/keras-hub/issues/1836) contains an overview of the project goals and our
current focus areas.

We follow
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Finding an issue

The fastest way to contribute it to find open issues that need an assignee. We
maintain two lists of github tags for contributors:

- [good first issue](https://github.com/keras-team/keras-hub/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22):
  a list of small, well defined issues for newcomers to the project.
- [contributions welcome](https://github.com/keras-team/keras-hub/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22):
  a larger list of issues that may range in complexity.

If you would like propose a new symbol or feature, please first review our
design guide and roadmap linked above, and open an issue to discuss. If you have
a specific design in mind, please include a Colab notebook showing the proposed
design in a end-to-end example. Keep in mind that design for a new feature or
use case may take longer than contributing to an open issue with a
vetted-design.

## Contributing code

Follow these steps to submit your code contribution.

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If your code change involves the fixing of a bug, please include a
[Colab](https://colab.research.google.com/) notebook that shows
how to reproduce the broken behavior.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a PR without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in section
"Setup environment".

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the master branch in
[keras-team/keras-hub](https://github.com/keras-team/keras-hub).

### Step 4. Sign the Contributor License Agreement

After creating the pull request, you will need to sign the Google CLA agreement.
The agreement can be found at
[https://cla.developers.google.com/clas](https://cla.developers.google.com/clas).

### Step 5. Code review

CI tests will automatically be run directly on your pull request. Their
status will be reported back via GitHub actions.

There may be several rounds of comments and code changes before the pull
request gets approved by the reviewer.

### Step 6. Merging

Once the pull request is approved, a team member will take care of merging.

## Setting up an Environment

Python 3.11 or later is required.

Setting up your KerasHub development environment requires you to fork the
KerasHub repository and clone it locally. With the
[GitHub CLI](https://github.com/cli/cli) installed, you can do this as follows:

```shell
gh repo fork keras-team/keras-hub --clone --remote
cd keras-hub
```

Next we must setup a python environment with the correct dependencies. We
recommend using `conda` to set up a base environment, and `pip` to install
python packages from PyPI. The exact method will depend on your OS.

**Note**: Be careful not to use mix pre-packaged tensorflow and jax libraries in
`conda` with PyPI packages from `pip`. We recommend pulling _all_ KerasHub
dependencies via `pip` as described below.

### Linux (recommended)

For developing and unit testing the library, a CPU-only environment is often
sufficient. For any training or inference with the library, you will quickly
want accelerator support. The easiest way to get GPU support across all of our
backends is to set up a few different python environements and pull in all cuda
dependencies via `pip`.

The shell snippet below will install four conda environments: `keras-hub-cpu`,
`keras-hub-jax`, `keras-hub-torch`, and `keras-hub-tensorflow`. The cpu
environement supports all backends without cuda, and each backend environement
has cuda support.

```shell
conda create -y -n keras-hub-cpu python=3.11
conda activate keras-hub-cpu
pip install -r requirements.txt  # install deps
pip install -e .  # install keras-hub

for backend in "jax" "torch" "tensorflow"; do
    conda create -y -n keras-hub-${backend} python=3.11
    conda activate keras-hub-${backend}
    pip install -r requirements-${backend}-cuda.txt  # install deps
    pip install -e .  # install keras-hub
done
```

To activate the jax environment and set keras to use jax, run:

```shell
conda activate keras-hub-jax && export KERAS_BACKEND=jax
```

### MacOS

`tensorflow-text` does not release precompiled binaries for MacOS M-series
chips, though the library does support building from source on MacOS.

We strongly recommend a Linux development environment for an easy contribution
experience. To build a dev environement from scratch on MacOS, see the following
guides:

- https://developer.apple.com/metal/tensorflow-plugin/
- https://github.com/tensorflow/text

### Windows

For the best experience developing on windows, please install
[WSL](https://learn.microsoft.com/en-us/windows/wsl/install), and proceed with
the linux installation instruction above.

To run the format and lint scripts, make sure you clone the repo with Linux
style line endings and change any line separator settings in your editor.
This is automatically done if you clone using git inside WSL.

Note that will not support Windows Shell/PowerShell for any scripts in this
repository.

## Testing changes

KerasHub is tested using [PyTest](https://docs.pytest.org/en/6.2.x/).

### Run a test file

To run a test file, run `pytest path/to/file` from the root directory of the
repository.

### Run a single test case

To run a single test, you can use `-k=<your_regex>`
to use regular expression to match the test you want to run. For example, you
can use the following command to run all the tests in `import_test.py`
whose names contain `import`:

```shell
pytest keras_hub/integration_tests/import_test.py -k="import"
```

### Run the full test suite

You can run the default testing suite by simply invoking pytest:

```shell
pytest
```

We annotate tests that are slower or require a network connection as "large",
and by default `pytest` will skip these tests. We run large tests continuously
on GCP. You can specify these by running:

```shell
pytest --run_large
```

Finally, for tests that are very slow and resource intensive (e.g. downloading
a 5GB checkpoint), we use an "extra_large" annotation and do not run them
continuously at all. You can specify these by running:

```shell
pytest --run_extra_large
```

When running "extra_large" tests, we recommend also specify a specific test file
so you aren't waiting around forever!

## Generating public API and formatting the code

For the first time you are setting up the repo, please run `pre-commit install`.
Note that this needs to be done only once at the beginning.

Now, whenever you run `git commit -m "<message>"`, three things are
automatically done:

- Public API generation
- Code formatting
- Code linting

If there's any error, the commit will not go through. Please fix the error (
most of the times, the error is fixed automatically by the formatter/linter) and
re-run the following:

```shell
git add .
git commit -m "<message>" # This will not get logged as a duplicate commit.
```

In case you want to run the above manually on all files, you can do the
following:

```shell
pre-commit run --all-files
```

KerasHub uses [Ruff](https://docs.astral.sh/ruff/) to format the code.

## Co-working with the Gemini CLI

Let's accelerate the development with Gemini CLI.

### Installation

Please refer to the Installation section at [https://github.com/google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli).

### Using the Gemini CLI

Start the CLI and analyze the project structure.

```shell
gemini

# In the CLI.
/init
```

After running this, a `GEMINI.md` file will be generated in the project root. This file contains the project context that the Gemini CLI will use for subsequent tasks.

### Adding models

Taking `DINOV3` as a concrete example, you can instruct the CLI to help implement a new model by providing clear references and local context.

```shell
# In the CLI.
Add `DINOV3Backbone` at @keras_hub/src/models/dinov3. Refer to the implementation on HF here: https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py and consider the existing implementation of `dinov2` at @keras_hub/src/models/dinov2 for guidance.
```

After the CLI generation, you should get some initial implementation for the model. Feel free to review and refine the code as needed.

Next, let's instruct the CLI to construct a numerical validation test to ensure the implementation is correct. Before running this step, make sure you have installed the `transformers` library and have access to the `facebook/dinov3-*` presets.

```shell
# In the CLI.
Create a numerical validation script `check_dinov3.py` for `DINOV3Backbone` at project root. Use the HF preset `facebook/dinov3-vits16-pretrain-lvd1689m` as a reference for the expected outputs. Remember to port the weights from HF to `DINOV3Backbone` within the script and refer to the existing implementation here: @keras_hub/src/utils/transformers/convert_dinov2.py
```

Now, instruct the CLI to run the script and correct any errors. If you are working within Conda environments, be sure to also instruct the CLI to use the appropriate environment for execution.

```shell
# In the CLI.
Run @check_dinov3.py by `KERAS_BACKEND=jax conda run -n keras-hub-jax python check_dinov3.py`. Fix any errors encountered during execution.
```

During this phase, human intervention is often necessary. You will need to carefully review the CLI's modifications and provide guidance or even handcraft some details that the tool failed to implement correctly.

Once you successfully complete the step above, you can now proceed to add the conversion script and unit tests for the `DINOV3Backbone`.

```shell
# In the CLI.
Create the conversion script `convert_dinov3.py` at @keras_hub/src/utils/transformers/convert_dinov3.py. Refer to the existing @keras_hub/src/utils/transformers/convert_dinov2.py at the same location for guidance.
```

```shell
# In the CLI.
Create unit tests for `DINOV3Backbone` at @keras_hub/src/models/dinov3. Refer to the existing tests for `DINOV2Backbone` at @keras_hub/src/models/dinov2/dinov2_backbone_test.py for guidance.
```

If you successfully run through all these steps, congratulations! You have now successfully added a new model to KerasHub through effective co-working with the Gemini CLI.

## Using the Model Porter Tool

The Model Porter tool automates the process of porting models from Hugging Face to KerasHub. It analyzes the KerasHub structure, understands file dependencies, and generates files in the correct order using an LLM (Gemini, Claude, or OpenAI).

### Usage

To use the tool, run the [`tools/model_porter.py`](tools/model_porter.py) script. You need to provide the target model name, a reference model name (an existing KerasHub model), your API key, and an output directory.

```shell
# Use Gemini (default)
python tools/model_porter.py --model_name <target_model> --reference_model <reference_model> --api_key <YOUR_API_KEY> --output_dir <output_dir>

# Use Claude
python tools/model_porter.py --model_name <target_model> --reference_model <reference_model> --api_key <YOUR_API_KEY> --api_provider claude --output_dir <output_dir>
```

For example, to port `qwen3` using `mixtral` as a reference:

```shell
python tools/model_porter.py --model_name qwen3 --reference_model mixtral --api_key $GEMINI_API_KEY --output_dir qwen3
```
