# OPSD Self-Contrast Project

## Overview
Repurposing On-Policy Self-Distillation (OPSD) signals as zero-training error localization for mathematical reasoning. The core idea: same frozen LLM, two forward passes (with/without privileged reference info), token-level distributional contrast localizes reasoning errors at the step level.

## Project Structure
```
opsd/
├── pilot_self_judge.py      # Pilot experiment: token-level analysis on GSM8K
├── exp_processbench.py      # Full ProcessBench experiment: step-level analysis
├── gsm8k_test.jsonl         # GSM8K test set (1,319 examples)
├── data/processbench/       # ProcessBench dataset (Qwen/ProcessBench)
│   ├── gsm8k.json           # 400 examples
│   ├── math.json            # 1,000 examples
│   ├── olympiadbench.json   # 1,000 examples
│   └── omnimath.json        # 1,000 examples
├── results/                 # Experiment outputs
│   ├── pilot/               # 100-sample GSM8K pilot
│   ├── processbench/        # GSM8K+MATH 1,400 samples
│   └── pb_3b_all/           # All 4 sources, 3,400 samples
├── run_*.sh                 # Slurm job scripts (HPC3, account=d_yings_team)
└── findings_0609.md         # Current findings report
```

## HPC Setup
- **HPC3**: Slurm cluster, H100 80GB GPUs, account `d_yings_team`, partition `acd_u`
- **Conda env**: `fgsd` (torch 2.6.0+cu124, transformers 4.57.6)
- **Models**: `~/data/models_dl/Qwen2.5-{0.5B,1.5B,3B,14B}-Instruct`
- **Note**: 14B model shard 5 is corrupted (25MB vs ~3.8GB), needs re-download

## Key Findings (June 9, 2026)
- TokenIncompatibility (TI) is the effective error signal, not InfoGain (KL)
- 3B zero-training: F1 42.3 on ProcessBench, beats most 7B trained PRMs
- Scales inversely with difficulty (better on hard problems, opposite of PRMs)
- Targeted z (answer only) >> exhaustive z (full solution) for localization
