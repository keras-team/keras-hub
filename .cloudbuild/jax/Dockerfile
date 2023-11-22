FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt -y update
RUN apt install -y python3 python3-pip git
RUN git clone https://github.com/keras-team/keras-nlp.git
RUN pip install -r keras-nlp/requirements-jax-cuda.txt
