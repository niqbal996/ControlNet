srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p A100-80GB --mem=50000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/ControlNet:/home/iqbal/ControlNet \
  --container-image=/netscratch/naeem/controlnet_22.05.sqsh  \
  --container-save=/netscratch/naeem/controlnet_22.05.sqsh \
  --time=00-02:00 \
  --pty /bin/bash
