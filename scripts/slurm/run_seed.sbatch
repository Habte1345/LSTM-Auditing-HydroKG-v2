#!/bin/bash
#SBATCH --job-name=hydrokg
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
# One array task per seed. Edit CAMELS_ROOT and EXPERIMENT; resubmitting resumes from the
# last completed step.
set -euo pipefail
CAMELS_ROOT=/path/to/CAMELS_US
EXPERIMENT=configs/experiments/temporal.yaml
SEEDS=(658666 111 222)
mkdir -p logs
python -m hydrokg.pipeline.cli run --experiment "$EXPERIMENT" --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" \
  --set paths.camels_root="$CAMELS_ROOT" experiment.audit_workers="$SLURM_CPUS_PER_TASK"
