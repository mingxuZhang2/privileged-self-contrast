#!/bin/bash
#SBATCH --job-name=opsd_pilot
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err

mkdir -p logs results

CONDA_BASE=~/miniconda3
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate fgsd

echo "=== Environment ==="
hostname
nvidia-smi
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=== Running pilot: 100 samples ==="
python pilot_self_judge.py \
    --model_path ~/data/models_dl/Qwen2.5-3B-Instruct \
    --data_path gsm8k_test.jsonl \
    --output_dir results/pilot \
    --n_samples 100 \
    --max_tokens 512

echo "=== Done ==="
