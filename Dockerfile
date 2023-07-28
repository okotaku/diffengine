FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    vim \
    libgl1-mesa-dev
ENV FORCE_CUDA="1"

# Install python package.
WORKDIR /modules
COPY ./ /modules
RUN pip install --upgrade pip && \
    pip install --no-cache-dir openmim==0.3.6 && \
    pip install .

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
