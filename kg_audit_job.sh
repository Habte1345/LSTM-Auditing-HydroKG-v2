#!/bin/bash
#SBATCH -J hydroKG
#SBATCH -p main
#SBATCH --qos main
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o logs/hydrokg-reaudit_%j.out
#SBATCH -e logs/hydrokg-reaudit_%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate BUDYKO-dev
cd /bighome/hdagne1/LSTM-Auditing-HydroKG-v2
V1=/bighome/hdagne1/LSTM-Auditing-HydroKG
CAMELS_ROOT=$V1/external/HydroAuditToolFrameowrk/data/CAMELS_US
RUN=$V1/external/HydroAuditToolFrameowrk/runs/run_0305_2015_seed658666
python -m hydrokg.pipeline.cli reaudit --experiment configs/experiments/reaudit_v1.yaml \
  --predictions $RUN/lstm_seed658.p $V1/results/hydrokg_run2_raw_enhanced_predictions.p $V1/results/hydrokg_run2_enhanced_predictions.p \
  --labels baseline v1_finetuned v1_final \
  --run-dir $RUN --out results/reaudit_v1 \
  --set paths.camels_root="$CAMELS_ROOT" experiment.audit_workers=8