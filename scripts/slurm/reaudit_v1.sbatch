#!/bin/bash
#SBATCH --job-name=hydrokg-reaudit
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Re-audit the v1 predictions (baseline, fine-tuned before correction, final) with the
# revised rules. No training; separates the fine-tuning and correction effects.
set -euo pipefail
CAMELS_ROOT=/path/to/CAMELS_US
V1=/bighome/hdagne1/LSTM-Auditing-HydroKG
RUN=$V1/external/HydroAuditToolFrameowrk/runs/run_0305_2015_seed658666
mkdir -p logs
python -m hydrokg.pipeline.cli reaudit --experiment configs/experiments/reaudit_v1.yaml \
  --predictions $RUN/lstm_seed658.p $V1/results/hydrokg_run2_raw_enhanced_predictions.p $V1/results/hydrokg_run2_enhanced_predictions.p \
  --labels baseline v1_finetuned v1_final \
  --run-dir $RUN --out results/reaudit_v1 \
  --set paths.camels_root="$CAMELS_ROOT" experiment.audit_workers="$SLURM_CPUS_PER_TASK"
