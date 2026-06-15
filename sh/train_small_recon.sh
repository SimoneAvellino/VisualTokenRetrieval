#!/bin/bash
#SBATCH --job-name=small-recon
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.avellino02@gmail.com
#SBATCH --output=logs/job-%j.log

# Runs the SMALL moment-reconstructor ladder: trivial baselines (full_video_mean,
# mean_token) + ONE trained model (query -> fixed 16-token sequence, soft-DTW loss,
# temporal hard negatives). Produces the reconstruction + retrieval (Recall@1/@5)
# comparison figures. Each finished step is recorded in the summary JSON, so
# resubmitting after a time limit resumes from the first unfinished step.

export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "Start time  : $(date)"
echo "============================================"

cd /home/vllsmn02h03b202i/rosario/VisualTokenRetrieval

apptainer run --nv /shared/sifs/latest.sif \
    python -m src.training.ladder \
        --config experiments/configs/ladder_small_recon_cluster.yaml

echo "============================================"
echo "End time : $(date)"
echo "============================================"
