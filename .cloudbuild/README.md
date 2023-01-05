# KerasNLP Accelerators Testing

This `cloudbuild/` directory contains configurations for accelerators (GPU/TPU)
testing. Briefly, for each PR, it copies the PR's code to a base docker image
which contains KerasNLP dependencies to make a new docker image, and deploys the
new image to Google Kubernetes Engine cluster, then run all tests in
`keras_nlp/` via Google Cloud Build.

- `cloudbuild.yaml`: The cloud build configuration that specifies steps to run
  by cloud build.
- `Dockerfile`: The configuration to build the docker image for deployment.
- `requirements.txt`: Dependencies of KerasNLP.
- `unit_test_jobs.jsonnet`: Jsonnet config that tells GKE cluster to run all
  unit tests in `keras_nlp/`.

This test is powered by [ml-testing-accelerators](https://github.com/GoogleCloudPlatform/ml-testing-accelerators).

### Adding Test Dependencies

You must be authorized to run builds in the `keras-team-test` GCP project.
If you are not, please open a GitHub issue and ping a team member.
To authorize yourself with `keras-team-test`, run:

```bash
gcloud config set project keras-team-test
```

To add/update dependency for GPU tests:
- Add/update dependency to `requirements.txt`
- Create a `Dockerfile` with the following contents:

  ```
  FROM tensorflow/tensorflow:2.11.0-gpu
  RUN apt-get -y update
  RUN apt-get -y install git
  RUN git clone https://github.com/keras-team/keras-nlp.git
  RUN cd keras-nlp
  RUN pip install -r keras-nlp/requirements.txt
  ```

- Run the following command from the directory with your `Dockerfile`:

  ```
  gcloud builds submit --region=us-west1 --tag us-west1-docker.pkg.dev/keras-team-test/keras-nlp-test/keras-nlp-image:deps --timeout=30m
  ```

### Run TPU Testing

Because of the TPU capacity limit, we cannot set automatic TPU testing. To
trigger the TPU testing, run the following command:

```
gcloud builds submit --config .cloudbuild/tpu_cloudbuild.yaml . --project=keras-team-test
```