FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update

WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/Otnielush/character_training.git
RUN cd character_training && git submodule update --init --recursive

WORKDIR /app/character_training
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN apt-get install -y tmux nvtop htop

ENV HF_HOME="/workspace/hf"
ENV HF_HUB_CACHE="/workspace/hf"
ENV GPU_NUM=1

# Important paths
# dir for saving trained weights
WORKDIR /training_output
# dir for images to train on
WORKDIR /dataset
# dir for models  /workspace/hf
# dir for training config /workspace/config


WORKDIR /app/character_training
RUN ["chmod", "+x", "/app/character_training/start_training.sh"]
CMD ["/app/character_training/start_training.sh"]

# run container important params:
    # -e GPU_NUM=_number of GPUs, default 1
    # -v /dataset
    # -v /workspace/hf
    # -v /training_output
    # -v /workspace/config
