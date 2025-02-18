#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=4
workdir=./inference
output_dir=./visualization_outputs  # Define output directory

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/inference.py \
	 $workdir/svg_pointT.yaml  $workdir/best.pth --dist --out $output_dir
