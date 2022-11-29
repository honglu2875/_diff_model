#!/bin/bash
#SBATCH --job-name=diff0
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=6
#SBATCH --output=diff_%x_%j.out
#SBATCH --comment=elm

source /fsx/home-honglu/miniconda3/bin/activate
conda activate training

python3 make_diff_dataset.py -i diff_data -o hf_diff_dataset -th 45
