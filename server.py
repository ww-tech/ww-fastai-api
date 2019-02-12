import os
import sys
import logging
import json
from fastai.text import *
from custom_fastai_nlp import *
from flask import Flask, Response
from flask import request, abort
from flask import jsonify

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)

MODELS_PATH = os.environ.get('MODELS_PATH', None)
DATA_CLASS_NAME = os.environ.get('DATA_CLASS_NAME', None)
FINE_TUNED_ENC = os.environ.get('FINE_TUNED_ENC', None)
FINAL_LAYER_PATH = os.environ.get('FINAL_LAYER_PATH', None)
BS = int(os.environ.get('BS', '48'))
DROP_MULTI = float(os.environ.get('DROP_MULTI', '0.5'))
HEALTHCHECK  = os.environ.get('HEALTHCHECK', '/hc')
PORT  = int(os.environ.get('PORT', '8080'))

### GCP Flags
DOWNLOAD_MODELS = os.environ.get("DOWNLOAD_MODELS", None)
GOOGLE_SERVICE_ACCOUNT = os.environ.get("GOOGLE_SERVICE_ACCOUNT", None)
GOOGLE_PROJECT = os.environ.get('GOOGLE_PROJECT', None)
MODELS_BUCKET_PATH = os.environ.get('MODELS_BUCKET_PATH')

fastai_nlp = FastAI_API(MODELS_PATH, 
                   DATA_CLASS_NAME,
                   FINE_TUNED_ENC,
                   FINAL_LAYER_PATH,
                   bs=BS,
                   drop_multi=DROP_MULTI,
                   download_gcs_models=DOWNLOAD_MODELS,
                   google_service_account=GOOGLE_SERVICE_ACCOUNT,
                   google_project=GOOGLE_PROJECT,
                   models_bucket_path=MODELS_BUCKET_PATH)
model = fastai_nlp.create_learner(URLs.IMDB)

class HealthCheckFilter(logging.Filter):  
    def filter(self, record):  
        return HEALTHCHECK not in record.getMessage()   

for handler in logging.root.handlers:  
    handler.addFilter(HealthCheckFilter())

@app.errorhandler(400)
def handle_bad_response_error(error):
    return Response('Must pass in JSON with {"message": "the string to predict"}')

@app.errorhandler(Exception)
def handle_global_error(error):
    return Response("Internal Server Error")

@app.route(HEALTHCHECK)
def hello():
    return 'Serving: {}'.format(MODELS_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json == False:
        abort(400)
    message = request.get_json().get('message', None)
    if message == None:
        abort(400)
    
    prediction = model.predict(message)
    data = {"message": message, "classification": str(prediction[0]), "confidence": round(prediction[2].data.tolist()[1], 3)}
    app.logger.info(json.dumps(data))
    return jsonify(data)

app.run(host='0.0.0.0', port=PORT)