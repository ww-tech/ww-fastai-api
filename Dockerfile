FROM python:3.6-slim
RUN apt-get update && \
    apt-get install -y gcc python-dev python-setuptools && \
    apt install -y curl && \
    pip install http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl && \
    pip install fastai && \
    pip uninstall crcmod && \
    pip install -U crcmod && \
    pip install -U Flask && \
    curl -sSL https://sdk.cloud.google.com | bash
RUN mkdir /opt/api/ && \
    mkdir /opt/api/models/ && \
    mkdir /opt/api/models/pos && \
    mkdir /opt/api/models/neg && \
    mkdir /opt/api/models/unsup && \
    mkdir /opt/api/models/tmp_class
ENV PATH $PATH:/root/google-cloud-sdk/bin
WORKDIR /opt/api/
ADD server.py server.py
ADD custom_fastai_learner.py custom_fastai_learner.py 
ADD custom_fastai_nlp.py custom_fastai_nlp.py
CMD ["python3", "server.py"]
