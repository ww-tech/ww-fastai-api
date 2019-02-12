# FastAI API

A Generic docker image to run fastai models. Currently only supports fastai.text models.

## NLP

You can run this with either of the docker-compose files. 

`docker-compose-local.yml` will pass in local models into your docker image. This speeds up local development with the models so you can quickly test how things are working. 

`docker-compose.yml` will download models from a GCS bucket before the API starts. This is how the application should be run in production. This allows the image to generically run any model, just by changing the environment variables.


### Build

```
docker build -t fastai_api .
```

### Usage

#### Environment Variables

```
GOOGLE_SERVICE_ACCOUNT: base64 encoded version of the google service account, that allows access to the gcs bucket path defined in the docker-compose file.
GOOGLE_PROJECT: The google project the models live in.
MODELS_BUCKET_PATH: The full bucket path the models live in.
DOWNLOAD_MODELS: "true" if you want to dowload models from gcs on startup. This is for production use.
FINE_TUNED_ENC: name of fine tuned encoder
FINAL_LAYER_PATH: name of final layer
DATA_CLASS_NAME: name of folder that holds the data models
MODELS_PATH: path to write the models in the container
```

#### Example Compose File

```
version: "2"
services:
  fastai-api:
    image: fastai_api
    ports:
      - "8080:8080"
    environment:
      GOOGLE_SERVICE_ACCOUNT: ${GOOGLE_SERVICE_ACCOUNT}
      DOWNLOAD_MODELS: "true"
      FINE_TUNED_ENC: "fine_tuned_enc.pth"
      FINAL_LAYER_PATH: "final.pth"
      DATA_CLASS_NAME: "tmp_class"
      MODELS_PATH: "/opt/api/models/"
      GOOGLE_PROJECT: "${GOOGLE_PROJECT}"
      MODELS_BUCKET_PATH: "${MODELS_BUCKET_PATH}"
```

docker-compose up
curl -s -H"Content-Type: application/json" localhost:8080/predict -d '{"message": "I love open source"}'
