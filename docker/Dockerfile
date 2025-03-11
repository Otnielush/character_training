FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update

WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

WORKDIR /app/ai-toolkit
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN apt-get install -y tmux nvtop htop

ENV HF_HOME="/workspace/hf"
ENV HF_HUB_CACHE="/workspace/hf"
ENV $Gpu_Num=1

# mask workspace
# dir for dataset and save weights, by default exists for runpod
# RUN mkdir /workspace


WORKDIR /app/ai-toolkit
CMD ["python otniel_scripts/caption_with_florence-2.py /workspace/dataset --output_dir /dataset",
     "accelerate launch --num_processes $Gpu_Num --mixed_precision bf16 --num_cpu_threads_per_process 2 run.py /workspace/config/train_config_h100.yaml"]