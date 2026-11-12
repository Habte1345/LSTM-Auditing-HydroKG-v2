#!/bin/bash
#SBATCH -J hydroKG
#SBATCH -p main
#SBATCH --qos main
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/hydrokg-pilot_%j.out
#SBATCH -e logs/hydrokg-pilot_%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate BUDYKO-dev
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /bighome/hdagne1/LSTM-Auditing-HydroKG-v2
V1=/bighome/hdagne1/LSTM-Auditing-HydroKG
CAMELS_ROOT=$V1/external/HydroAuditToolFrameowrk/data/CAMELS_US
RUN=$V1/external/HydroAuditToolFrameowrk/runs/run_0305_2015_seed658666
python -m hydrokg.pipeline.cli run --experiment configs/experiments/temporal_original_baseline.yaml --seed 658666 \
  --set paths.camels_root="$CAMELS_ROOT" experiment.reuse_baseline_run_dir="$RUN" \
        experiment.audit_workers=$SLURM_CPUS_PER_TASK
