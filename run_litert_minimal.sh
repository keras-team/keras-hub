# Set environment to use local repositories
export KERAS_BACKEND=tensorflow
# export PYTHONPATH=/Users/hellorahul/Projects/keras:/Users/hellorahul/Projects/keras-hub:$PYTHONPATH
export PYTHONPATH=/Users/hellorahul/Projects/keras-hub:$PYTHONPATH
# Search for tests containing 'run_litert_export'
TEST_FILES=$(grep -rl "run_litert_export" keras_hub/src/models | grep "_test.py")

# Run only test_litert_export methods with verbose output
# Results are saved to 'litert_test_results.log'
pytest -vs -k test_litert_export $TEST_FILES 2>&1 | tee litert_test_results_tensorflow_pip_keras.log

export PYTHONPATH=/Users/hellorahul/Projects/keras:/Users/hellorahul/Projects/keras-hub:$PYTHONPATH
# export PYTHONPATH=/Users/hellorahul/Projects/keras-hub:$PYTHONPATH
# Search for tests containing 'run_litert_export'
TEST_FILES=$(grep -rl "run_litert_export" keras_hub/src/models | grep "_test.py")

# Run only test_litert_export methods with verbose output
# Results are saved to 'litert_test_results.log'
pytest -vs -k test_litert_export $TEST_FILES 2>&1 | tee litert_test_results_tensorflow_local_keras.log

export KERAS_BACKEND=torch
pytest -vs -k test_litert_export $TEST_FILES 2>&1 | tee litert_test_results_torch_local_keras.log