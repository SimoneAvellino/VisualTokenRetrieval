#!/bin/bash
#SBATCH --job-name=extract-tokens
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.avellino02@gmail.com
#SBATCH --output=logs/job-%j.log

mkdir -p logs

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "Start time  : $(date)"
echo "============================================"

cd /home/vllsmn02h03b202i/rosario/VisualTokenRetrieval

apptainer run --nv /shared/sifs/latest.sif \
    python -m src.datasets.extract_visual_tokens \
        --config experiments/configs/extract_cluster.yaml

echo "============================================"
echo "End time : $(date)"
echo "============================================"
