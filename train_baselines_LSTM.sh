#!/bin/bash
#SBATCH --job-name=hydroKG
#SBATCH --output=/bighome/hdagne1/LSTM-Auditing-HydroKG-v2/logs/baseline_%A_%a.out
#SBATCH --error=/bighome/hdagne1/LSTM-Auditing-HydroKG-v2/logs/baseline_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --exclude=uahpc-gpu001,uahpc-gpu002
#SBATCH --array=0-4
#SBATCH --requeue

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate BUDYKO-dev
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /bighome/hdagne1/LSTM-Auditing-HydroKG-v2

SEEDS=(658666 111 444 666 777)
CAMELS_ROOT=/bighome/hdagne1/LSTM-Auditing-HydroKG/external/HydroAuditToolFrameowrk/data/CAMELS_US

nvidia-smi --query-gpu=name --format=csv,noheader
python -m hydrokg.pipeline.cli run --experiment configs/experiments/temporal.yaml \
  --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} \
  --set paths.camels_root="$CAMELS_ROOT" lstm.device=cuda \
        experiment.arms=[] experiment.corrections=[none] \
        experiment.audit_workers=$SLURM_CPUS_PER_TASK