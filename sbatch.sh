#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/%j.out
#SBATCH --constraint="80gb|40gb"
#SBATCH --mail-user=tim.wiebelhaus@tu-berlin.de

# 1. copy the squashed dataset to the nodes /tmp
cp elec.sqfs /tmp/
# 2. Show that it copied!
ls /tmp/ | grep elec

# 3. see if container properly mounted
apptainer run --nv -B /tmp/elec.sqfs:/input-data/:image-src=/ python_container_cuda.sif sh -c "ls -l /input-data | wc -l"

# 4. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp/elec.sqfs:/input-data/:image-src=/ python_container_cuda.sif python iTransformer/hyperparameter_search.py --save_all
