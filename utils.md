rsync -av --progress \
 --exclude='.git/' \
 --exclude='.venv/' \
 --exclude='venv/' \
 --exclude='env/' \
 --exclude='_.egg-info/' \
 --exclude='**pycache**/' \
 --exclude='_.pyc' \
 --exclude='data/' \
 --exclude='figures/' \
 --exclude='figures_dry/' \
 --exclude='experiments/checkpoints/' \
 --exclude='experiments/logs/' \
 --exclude='experiments/results/' \
 --exclude='logs/' \
 --exclude='wandb/' \
 --exclude='\*.pth' \
 --exclude='.DS_Store' \
 ~/Desktop/VisualTokenRetrieval/ \
 vllsmn02h03b202i@gcluster.dmi.unict.it:/home/vllsmn02h03b202i/rosario/VisualTokenRetrieval/

tail -f $(ls -v logs/\* | tail -n 1)

sbatch sh/train_dry_run.sh

rsync -av --progress \
 vllsmn02h03b202i@gcluster.dmi.unict.it:/home/vllsmn02h03b202i/rosario/figures/ \
 ~/Desktop/VisualTokenRetrieval/figures/cluster/
