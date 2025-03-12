FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update

WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/Otnielush/character_training.git && \
    cd character_training && \
    git submodule update --init --recursive

WORKDIR /app/character_training
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN apt-get install -y tmux nvtop htop

ENV HF_HOME="/workspace/hf"
ENV HF_HUB_CACHE="/workspace/hf"
ENV $GPU_NUM=1

# mask workspace
# dir for dataset and save weights, by default exists for runpod
# RUN mkdir /workspace
# dir for saving trained weights
# RUN mkdir /training_output


WORKDIR /app/character_training
CMD ["python otniel_scripts/caption_with_florence-2.py /dataset --output_dir /train_dataset",
     "accelerate launch --num_processes $GPU_NUM --mixed_precision bf16 --num_cpu_threads_per_process 2 run.py /workspace/config/train_config_$GPU_NUMh100.yaml"]