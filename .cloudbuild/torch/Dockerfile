FROM nvidia/cuda:11.7.1-base-ubuntu20.04
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git
RUN git clone https://github.com/keras-team/keras-nlp.git
RUN cd keras-nlp
RUN pip install -r keras-nlp/requirements.txt
RUN pip install torch
