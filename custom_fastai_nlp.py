import os
import subprocess
import base64
from fastai.text import *
from custom_fastai_learner import *

class FastAI_API():
    def __init__(self, models_path, 
                       data_class_name,
                       fine_tuned_enc_name,
                       final_layer_name,
                       bs=48,
                       drop_multi=0.5,
                       download_gcs_models=False,
                       google_service_account=None,
                       google_project=None,
                       models_bucket_path=None):
        self.models_path = models_path
        self.data_class_name = data_class_name
        self.fine_tuned_enc_name = fine_tuned_enc_name
        self.final_layer_name = final_layer_name
        self.bs = bs
        self.drop_multi = drop_multi
        self.download_gcs_models = download_gcs_models
        self.google_service_account = google_service_account
        self.google_project = google_project
        self.models_bucket_path = models_bucket_path

        if self.download_gcs_models == "true":
            self.download_models()

    def download_models(self):

        # write service account to /tmp/config
        g_path = "/tmp/config"

        if self.google_service_account != None:
            f = open(g_path, "w")
            f.write(base64.b64decode(self.google_service_account).decode('utf-8'))
            f.close()
        
            # activate service account
            bashCommand = "gcloud auth activate-service-account --key-file {}".format(g_path)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
    
        # download files on startup
        bashCommand = "gsutil cp gs://{} {}".format(self.models_bucket_path + self.fine_tuned_enc_name, self.models_path + self.fine_tuned_enc_name)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        bashCommand = "gsutil -m cp -r gs://{} {}".format(self.models_bucket_path + self.data_class_name, self.models_path)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        bashCommand = "gsutil cp gs://{} {}".format(self.models_bucket_path + self.final_layer_name , self.models_path + self.final_layer_name)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        os.remove(g_path)
        
        if error != None:
            raise Exception(error)
    
    def create_learner(self, urls):
        path = untar_data(urls)
        data_class = TextClasDataBunch.load(self.models_path, self.data_class_name, bs=self.bs)
        learn = text_classifier_learner(data_class, drop_mult=self.drop_multi)
        learn.load_encoder(self.models_path + self.fine_tuned_enc_name.replace(".pth", ""))
        learn.load(self.models_path + self.final_layer_name.replace(".pth", ""))
        return learn