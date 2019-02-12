# FastAI API

A Generic docker image to run fastai models. Currently only supports fastai.text models.

## NLP

You can run this with either of the docker-compose images. 

docker-compose-local will pass in local models into your docker image. This speeds up local development with the models so you can quickly test how things are working. 

docker-compose will download models from a GCS bucket before the API starts. This is how the application should be run in production. This allows the image to generically run any model, just by changing the environment variables.

### Usage

```
# must have the correct $GOOGLE_SERVICE_ACCOUNT environment set that allows access to the gcs bucket path defined in the docker-compose file.
docker-compose up
curl -s -H"Content-Type: application/json" localhost:8080/predict -d '{"message": "I love open source"}'
