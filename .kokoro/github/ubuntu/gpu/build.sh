set -e

export KAGGLE_KEY="$(cat ${KOKORO_KEYSTORE_DIR}/73361_keras_kaggle_secret_key)"
export KAGGLE_USERNAME="$(cat ${KOKORO_KEYSTORE_DIR}/73361_keras_kaggle_username)"

if [[ -z "${KAGGLE_KEY}" ]]; then
   echo "KAGGLE_KEY is NOT set"
   exit 1
fi

if [[ -z "${KAGGLE_USERNAME}" ]]; then
   echo "KAGGLE_USERNAME is NOT set"
   exit 1
fi

set -x
cd "${KOKORO_ROOT}/"

export DEBIAN_FRONTEND=noninteractive
cd "${KOKORO_ROOT}/"

PYTHON_BINARY="/usr/bin/python3.10"

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate
# Check the python version
python --version
python3 --version

# setting the LD_LIBRARY_PATH manually is causing segmentation fault
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:"
# Check cuda
nvidia-smi
nvcc --version

cd "src/github/keras-hub"
pip install -U pip setuptools psutil

if [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off \
      --timeout 1000

elif [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   pip install -r requirements-jax-cuda.txt --progress-bar off --timeout 1000

elif [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   pip install -r requirements-torch-cuda.txt --progress-bar off --timeout 1000
fi


# Temporarily relax Config to allow 3.10 for GPU tests.
sed -i 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' pyproject.toml

pip install --no-deps -e "." --progress-bar off
pip install huggingface_hub

# Run Extra Large Tests for Continuous builds
if [ "${RUN_XLARGE:-0}" == "1" ]
then
   pytest keras_hub --check_gpu --run_large --run_extra_large \
      --cov=keras_hub
else
   pytest keras_hub --check_gpu --run_large \
      --cov=keras_hub
fi
