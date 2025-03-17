FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update

WORKDIR /app
ARG CACHEBUST=1

RUN git clone https://github.com/Otnielush/character_inference.git
RUN git clone https://github.com/Otnielush/character_training.git
RUN cd character_training && git submodule update --init --recursive

WORKDIR /app/character_training
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app/character_inference
RUN python3 -m pip install --no-cache-dir -r requirements_docker.txt

ENV TORCH_CUDA_ARCH_LIST=Turing
ENV PATH /opt/conda/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.2/compat/:/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

ENV CUDA_HOME='/usr/local/cuda'
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
WORKDIR /repos
RUN git clone https://github.com/aredden/torch-cublas-hgemm.git
WORKDIR /repos/torch-cublas-hgemm
RUN python3 -m pip install --no-build-isolation -U -v .


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
    # --gpus=all  - give access to gpus
    # -e GPU_NUM=_number of GPUs, default 1
    # -v /dataset
    # -v /workspace/hf
    # -v /training_output
    # -v /workspace/config
