FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    vim \
    libgl1-mesa-dev \
    zsh

# Zsh install
ENV SHELL /bin/zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell

# Install python package.
WORKDIR /diffengine
COPY ./ /diffengine
RUN pip install --upgrade pip && \
    pip install --no-cache-dir openmim==0.3.9 && \
    pip install .

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
