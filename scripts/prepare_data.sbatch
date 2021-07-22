#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --partition=global
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=prepare_data%j.log
#SBATCH --mem-per-cpu=1024M

module load singularity/3.2.1
singularity exec ./container/tensorflowUbuntu.img ./scripts/prepare_data.bash --nv -c
