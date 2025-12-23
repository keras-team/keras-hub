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
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository universe
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

# Diagnostics
echo "Checking available python versions..."
pyenv versions || echo "pyenv not found"
apt-cache policy python3.11 || echo "python3.11 not in apt cache"

if ! command -v python3.11 &> /dev/null; then
    echo "python3.11 not found, attempting install..."
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev || echo "apt install failed"
fi

PYTHON_BINARY="python3.11"
if [ -x "/usr/bin/python3.11" ]; then
    PYTHON_BINARY="/usr/bin/python3.11"
elif [ -x "$(which python3.11)" ]; then
    PYTHON_BINARY="$(which python3.11)"
fi

echo "Using PYTHON_BINARY: ${PYTHON_BINARY}"

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
