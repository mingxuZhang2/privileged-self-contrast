#!/bin/bash
#SBATCH --job-name=opsd_14b
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/14b_%j.out
#SBATCH --error=logs/14b_%j.err

mkdir -p logs results

CONDA_BASE=~/miniconda3
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate fgsd

export PYTHONUNBUFFERED=1

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Run: GSM8K + MATH, 14B model ==="
python exp_processbench.py \
    --model_path ~/data/models_dl/Qwen2.5-14B-Instruct \
    --data_files data/processbench/gsm8k.json data/processbench/math.json \
    --output_dir results/pb_14b_gsm_math

echo "=== Done ==="
