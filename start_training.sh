#!/bin/bash
python otniel_scripts/caption_with_florence-2.py /dataset --output_dir /train_dataset
accelerate launch --num_processes $GPU_NUM --mixed_precision bf16 --num_cpu_threads_per_process 2 run.py /workspace/config/train_config_${GPU_NUM}h100.yaml