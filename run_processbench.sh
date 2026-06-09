#!/bin/bash
#SBATCH --job-name=opsd_pb
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pb_%j.out
#SBATCH --error=logs/pb_%j.err

mkdir -p logs results

CONDA_BASE=~/miniconda3
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate fgsd

export PYTHONUNBUFFERED=1

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

echo "=== Running ProcessBench: GSM8K + MATH (1400 samples) ==="
python exp_processbench.py \
    --model_path ~/data/models_dl/Qwen2.5-3B-Instruct \
    --data_files data/processbench/gsm8k.json data/processbench/math.json \
    --output_dir results/processbench

echo "=== Done ==="
