FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN mkdir -p /vqvae
WORKDIR /vqvae
SHELL ["/bin/bash", "--login", "-c"]

# tzdata wants to perform interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# install python3
RUN apt-get update -qqy \
 && apt-get -y install software-properties-common \
                       build-essential \
                       curl \
                       wget \
                       python3-pip \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update -qqy \
 && apt-get install -y python3.10 python3.10-dev python3.10-distutils \
 && apt-get clean

# pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
#RUN pip install --no-cache-dir -U pip

# Use virtualenv to avoid system python usage
RUN python3.10 -m pip install virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m virtualenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements*.txt /vqvae/

RUN --mount=type=cache,target=/root/.cache/pip cd /vqvae && pip install -r requirements-pytorch.txt
RUN --mount=type=cache,target=/root/.cache/pip cd /vqvae && pip install -r requirements-extra.txt

COPY ../ /vqvae

