#!/bin/bash -l

#SBATCH --nodes 4
#SBATCH --mem 32G

#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --gres-flags enforce-binding

#SBATCH --time 4:00:00

# Set up my modules
module purge
module load gcc cuda python

source path_to_venv/bin/activate

python main.py