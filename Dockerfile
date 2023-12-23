FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN apt update -y && apt install -y \
    git tmux
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
RUN pip install --upgrade pip

# Install xformers
RUN pip install ninja
RUN export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6 9.0+PTX" MAX_JOBS=1 && \
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.20#egg=xformers

# Install DiffEngine
RUN pip install --no-cache-dir openmim==0.3.9 && \
    pip install . && \
    pip install pre-commit

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
