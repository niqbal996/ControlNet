srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p A100-IML --mem=50000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/ControlNet:/home/iqbal/ControlNet \
  --container-image=/netscratch/naeem/controlnet_22.05.sqsh \
  --mail-type=BEGIN --mail-user=naeem.iqbal@dfki.de --job-name=controlnet_pheno \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=controlnet_pheno \
  --container-workdir=/home/iqbal/ControlNet \
  --time=02-00:00 \
  bash ./scripts/finetune_controlnet.sh
