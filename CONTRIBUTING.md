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

Python 3.7 or later is required.

Setting up your KerasNLP development environment requires you to fork the
KerasNLP repository and clone it locally. With the
[GitHub CLI](https://github.com/cli/cli) installed, you can do this as follows:

```shell
gh repo fork keras-team/keras-nlp --clone --remote
cd keras-nlp
```

Next we must setup a python environment with the correct dependencies. We
recommend using `conda` to install tensorflow dependencies (such as CUDA), and
`pip` to install python packages from PyPI. The exact method will depend on your
OS.

**Note**: Please be careful not to use the `tensorflow` pre-packaged with conda,
which is incompatible with `tensorflow-text` on PyPi, and follow the
instructions below.

### Linux (recommended)

To setup a complete environment with TensorFlow, a local install of keras-nlp,
and all development tools, run the following or adapt it to suit your needs.

```shell
# Create and activate conda environment.
conda create -n keras-nlp python=3.9
conda activate keras-nlp

# The following can be omitted if GPU support is not required.
conda install -c conda-forge cudatoolkit-dev=11.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e "."
```

### MacOS

⚠️⚠️⚠️ MacOS binaries are for the M1 architecture are not currently available from
official sources. You can try experimental development workflow leveraging the
[tensorflow metal plugin](https://developer.apple.com/metal/tensorflow-plugin/)
and a [community maintained build](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon)
of `tensorflow-text`. These binaries are not provided by Google, so proceed at
your own risk.

#### Experimental instructions for Arm (M1)

```shell
# Create and activate conda environment.
conda create -n keras-nlp python=3.9
conda activate keras-nlp

# Install dependencies.
conda install -c apple tensorflow-deps=2.9
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-m1.txt
python -m pip install -e "."
```

#### Instructions for x86 (Intel)

```shell
# Create and activate conda environment.
conda create -n keras-nlp python=3.9
conda activate keras-nlp

# Install dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e "."
```

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
