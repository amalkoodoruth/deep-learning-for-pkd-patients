#!/bin/bash
#SBATCH --account=def-alam1
#SBATCH --job-name='liver_seg'
#SBATCH --mail-user=amal.koodoruth@mail.mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=amal_output.out
#SBATCH --error=amal_error.err
#SBATCH --time=08:00:00 # increase as needed
#SBATCH --mem=G
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
nvidia-smi
module load python/3.7
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python train.py