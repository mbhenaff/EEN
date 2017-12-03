#!/bin/bash 
#
#SBATCH --job-name=VAE_${1}
#SBATCH --output=een.out
#SBATCH --error=een.err
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mbh305@nyu.edu

module load python
srun python train_vae.py -task $1 -n_latent $2 -lambda_kl $3 -lrt $4
