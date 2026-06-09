#!/bin/bash
#SBATCH --job-name=opsd_full
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

mkdir -p logs results

CONDA_BASE=~/miniconda3
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate fgsd

export PYTHONUNBUFFERED=1

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Run 1: All 4 sources, 3B model ==="
python exp_processbench.py \
    --model_path ~/data/models_dl/Qwen2.5-3B-Instruct \
    --data_files data/processbench/gsm8k.json data/processbench/math.json data/processbench/olympiadbench.json data/processbench/omnimath.json \
    --output_dir results/pb_3b_all

echo "=== Run 1 Done ==="
