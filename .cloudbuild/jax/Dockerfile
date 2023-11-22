FROM nvidia/cuda:12.2.0-base-ubuntu22.04
RUN apt-get -y update
RUN apt-get install -y python3 python3-pip git
RUN git clone https://github.com/keras-team/keras-nlp.git
RUN cd keras-nlp
RUN pip install -r keras-nlp/requirements-jax-cuda.txt
