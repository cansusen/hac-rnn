#!/bin/bash
#SBATCH -N 1                      # number of nodes
#SBATCH -n 16                     # number of cores
#SBATCH --mem=64G                 # memory pool for all cores
#SBATCH -t 0-5:00                 # time (D-HH:MM)
#SBATCH --gres=gpu:0              # number of GPU
#SBATCH -o slurm-skl-output       # STDOUT
#SBATCH -e slurm-skl-error        # STDERR

################################ 
################################

python run_svm_logistic.py
#python cui_to_bow.py 
