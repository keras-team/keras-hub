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

PYTHON_BINARY="/usr/bin/python3.9"

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate
# Check the python version
python --version
python3 --version

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:"
# Check cuda
nvidia-smi
nvcc --version

cd "src/github/keras-nlp"
pip install -U pip setuptools psutil

if [ "${KERAS2:-0}" == "1" ]
then
   echo "Keras2 detected."
   pip install -r requirements-common.txt --progress-bar off
   pip install tensorflow-text==2.15 tensorflow[and-cuda]~=2.15 keras-core

elif [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off

elif [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   pip install -r requirements-jax-cuda.txt --progress-bar off

elif [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   pip install -r requirements-torch-cuda.txt --progress-bar off
fi

pip install --no-deps -e "." --progress-bar off

# Run Extra Large Tests for Continuous builds
if [ "${RUN_XLARGE:-0}" == "1" ]
then
   pytest keras_nlp --check_gpu --run_large --run_extra_large \
      --cov=keras-nlp
else
   pytest keras_nlp --check_gpu --run_large \
      --cov=keras-nlp
fi