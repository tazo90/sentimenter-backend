# pull official base image
FROM python:3.8.1-slim

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# copy requirements file
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install gcc python3-dev -y \
    # && apt-get clean \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt
    # && rm -rf /root/.cache/pip

# copy project
COPY . /app/