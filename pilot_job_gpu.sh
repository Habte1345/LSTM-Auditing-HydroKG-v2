#!/bin/bash
#SBATCH --job-name=hydroKG
#SBATCH --output=/bighome/hdagne1/LSTM-Auditing-HydroKG-v2/logs/hydrokg-pilot_%j.out
#SBATCH --error=/bighome/hdagne1/LSTM-Auditing-HydroKG-v2/logs/hydrokg-pilot_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --requeue
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate BUDYKO-dev
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /bighome/hdagne1/LSTM-Auditing-HydroKG-v2
V1=/bighome/hdagne1/LSTM-Auditing-HydroKG
CAMELS_ROOT=$V1/external/HydroAuditToolFrameowrk/data/CAMELS_US
RUN=$V1/external/HydroAuditToolFrameowrk/runs/run_0305_2015_seed658666
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -m hydrokg.pipeline.cli run --experiment configs/experiments/temporal_original_baseline.yaml --seed 658666 \
  --set paths.camels_root="$CAMELS_ROOT" experiment.reuse_baseline_run_dir="$RUN" \
        experiment.audit_workers=$SLURM_CPUS_PER_TASK lstm.device=cuda
