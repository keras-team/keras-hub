# Contribution guide

## How to contribute code

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

## Setup environment

Python 3.7 or later is required.

Setting up your KerasNLP development environment requires you to fork the
KerasNLP repository, clone the repository, create a virtual environment, and 
install dependencies.

You can achieve this by running the following commands:

```shell
gh repo fork keras-team/keras-nlp --clone --remote
cd keras-nlp
python -m venv ~/keras-nlp-venv
source ~/keras-nlp-venv/bin/activate
pip install -e ".[tests]"
```

The first line relies on having an installation of
[the GitHub CLI](https://github.com/cli/cli).

Following these commands you should be able to run the tests using
`pytest keras_nlp`. Please report any issues running tests following these
steps.

## Run tests

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

### Run all tests

You can run the unit tests for KerasNLP by running:

```shell
pytest keras_nlp/
```

## Formatting the Code

We use `flake8`, `isort` and `black` for code formatting.  You can run
the following commands manually every time you want to format your code:

- Run `shell/format.sh` to format your code
- Run `shell/lint.sh` to check the result.

If after running these the CI flow is still failing, try updating `flake8`,
`isort` and `black`. This can be done by running `pip install --upgrade black`,
`pip install --upgrade flake8`, and `pip install --upgrade isort`.

Note that if you are using Windows Subsystem for Linux (WSL), make sure you 
clone the repo with Linux style LF line endings before running the format
or lint scripts. This is automatically done if you clone using git inside WSL.
If there is conflict due to the line endings you might see an error
like - `: invalid option`

## Community Guidelines

This project follows 
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
