# pull official base image
FROM tensorflow/tensorflow:2.1.0-py3

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# tensorflow allocation memory fix
ENV TF_CPP_MIN_LOG_LEVEL 3

# copy requirements file
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install gcc python3-dev python3-pip -y \
    # && apt-get clean \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt
# && rm -rf /root/.cache/pip

# install spacy en model & vader_lexicon
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader vader_lexicon

# copy project
COPY . /app/