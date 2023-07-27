FROM tensorflow/tensorflow:2.13.0-gpu
RUN apt-get -y update
RUN apt-get -y install git
RUN git clone https://github.com/keras-team/keras-nlp.git
RUN cd keras-nlp
RUN pip install -r keras-nlp/requirements.txt
