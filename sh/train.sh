#!/bin/bash
#SBATCH --job-name=train-query-gt
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.avellino02@gmail.com
#SBATCH --output=logs/job-%j.log

export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/home/vllsmn02h03b202i/rosario/checkpoints_query_gt/wandb_logs
export WANDB_MODE=offline

mkdir -p logs
mkdir -p $WANDB_DIR

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "Start time  : $(date)"
echo "============================================"

cd /home/vllsmn02h03b202i/rosario/VisualTokenRetrieval

apptainer run --nv /shared/sifs/latest.sif \
    python -m src.training.train \
        --config experiments/configs/train_cluster.yaml

echo "============================================"
echo "End time : $(date)"
echo "============================================"
