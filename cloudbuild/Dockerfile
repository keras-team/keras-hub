# Creates environment to run unit tests with current nightly image.
FROM tensorflow/tensorflow:latest-gpu
COPY . /kerasnlp
WORKDIR /kerasnlp
RUN apt-get -y update
RUN apt-get -y install git
RUN pip install -r cloudbuild/requirements.txt
CMD ["./cloudbuild/quickstart.sh"]