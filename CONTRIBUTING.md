# Contribution guide

KerasNLP is an actively growing project and community! We would love for you
to get involved. Below are instructions for how to plug into KerasNLP
development.

## Background reading

Before contributing code, please review our [Style Guide](STYLE_GUIDE.md) and
[API Design Guide](API_DESIGN_GUIDE.md).

Our [Roadmap](ROADMAP.md) contains an overview of the project goals and our
current focus areas.

We follow
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Finding an issue

The fastest way to contribute it to find open issues that need an assignee. We
maintain two lists of github tags for contributors:

 - [good first issue](https://github.com/keras-team/keras-nlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22):
   a list of small, well defined issues for newcomers to the project.
 - [contributions welcome](https://github.com/keras-team/keras-nlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22):
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
[keras-team/keras-nlp](https://github.com/keras-team/keras-nlp).

### Step 4. Sign the Contributor License Agreement

After creating the pull request, you will need to sign the Google CLA agreement.
The agreement can be found at
[https://cla.developers.google.com/clas](https://cla.developers.google.com/clas).

### Step 5. Code review

CI tests will automatically be run directly on your pull request.  Their
status will be reported back via GitHub actions.

There may be several rounds of comments and code changes before the pull
request gets approved by the reviewer.

### Step 6. Merging

Once the pull request is approved, a team member will take care of merging.

## Setting up an Environment

Python 3.9 or later is required.

Setting up your KerasNLP development environment requires you to fork the
KerasNLP repository and clone it locally. With the
[GitHub CLI](https://github.com/cli/cli) installed, you can do this as follows:

```shell
gh repo fork keras-team/keras-nlp --clone --remote
cd keras-nlp
```

Next we must setup a python environment with the correct dependencies. We
recommend using `conda` to set up a base environment, and `pip` to install
python packages from PyPI. The exact method will depend on your OS.

**Note**: Be careful not to use mix pre-packaged tensorflow and jax libraries in
`conda` with PyPI packages from `pip`. We recommend pulling *all* KerasNLP
dependencies via `pip` as described below.

### Linux (recommended)

For developing and unit testing the library, a CPU-only environment is often
sufficient. For any training or inference with the library, you will quickly
want accelerator support. The easiest way to get GPU support across all of our
backends is to set up a few different python environements and pull in all cuda
dependencies via `pip`.

The shell snippet below will install four conda environments: `keras-nlp-cpu`,
`keras-nlp-jax`, `keras-nlp-torch`, and `keras-nlp-tensorflow`. The cpu
environement supports all backends without cuda, and each backend environement
has cuda support.

```shell
conda create -y -n keras-nlp-cpu python=3.10
conda activate keras-nlp-cpu
pip install -r requirements.txt  # install deps
pip install -e .  # install keras-nlp

for backend in "jax" "torch" "tensorflow"; do
    conda create -y -n keras-nlp-${backend} python=3.10
    conda activate keras-nlp-${backend}
    pip install -r requirements-${backend}-cuda.txt  # install deps
    pip install -e .  # install keras-nlp
done
```

To activate the jax environment and set keras to use jax, run:

```shell
conda activate keras-nlp-jax && export KERAS_BACKEND=jax
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

## Update Public API

Run API generation script when creating PRs that update `keras_nlp_export`
public APIs. Add the files changed in `keras_nlp/api` to the same PR.

```
./shell/api_gen.sh
```

## Testing changes

KerasNLP is tested using [PyTest](https://docs.pytest.org/en/6.2.x/).

### Run a test file

To run a test file, run `pytest path/to/file` from the root directory of the
repository.

### Run a single test case

To run a single test, you can use `-k=<your_regex>`
to use regular expression to match the test you want to run. For example, you
can use the following command to run all the tests in `import_test.py`
whose names contain `import`:

```shell
pytest keras_nlp/keras_nlp/integration_tests/import_test.py -k="import"
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

## Formatting Code

We use `flake8`, `isort` and `black` for code formatting.  You can run
the following commands manually every time you want to format your code:

- Run `shell/format.sh` to format your code
- Run `shell/lint.sh` to check the result.

If after running these the CI flow is still failing, try updating `flake8`,
`isort` and `black`. This can be done by running `pip install --upgrade black`,
`pip install --upgrade flake8`, and `pip install --upgrade isort`.
