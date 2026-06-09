#!/bin/bash
#SBATCH --job-name=opsd_noleak
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/noleak_%j.out
#SBATCH --error=logs/noleak_%j.err

mkdir -p logs results

CONDA_BASE=~/miniconda3
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate fgsd

export PYTHONUNBUFFERED=1

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Non-leaky experiments: GSM8K + MATH ==="
python exp_nonleaky.py \
    --model_path ~/data/models_dl/Qwen2.5-3B-Instruct \
    --data_files data/processbench/gsm8k.json data/processbench/math.json \
    --output_dir results/nonleaky

echo "=== Done ==="
