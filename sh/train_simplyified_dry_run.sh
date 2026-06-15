#!/bin/bash
#SBATCH --job-name=mean-dry
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.avellino02@gmail.com
#SBATCH --output=logs/job-%j.log

# DRY RUN of the SIMPLIFIED mean-token ladder: caps epochs/batches so the whole
# pipeline (baselines + mean model + loss curves + comparison charts) runs
# end-to-end in minutes. Use this to verify the wiring on the cluster before
# launching the real run with train_simplyified.sh. Artifacts go to *_dry dirs
# and never touch the real ones.

export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID  (DRY RUN)"
echo "Node        : $SLURM_NODELIST"
echo "Start time  : $(date)"
echo "============================================"

cd /home/vllsmn02h03b202i/rosario/VisualTokenRetrieval

apptainer run --nv /shared/sifs/latest.sif \
    python -m src.training.ladder \
        --config experiments/configs/ladder_simplified_cluster_dry.yaml

echo "============================================"
echo "End time : $(date)"
echo "============================================"
