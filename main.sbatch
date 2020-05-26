#!/bin/bash
#SBATCH -p long
#SBATCH -N 1                      # number of nodes
#SBATCH -n 1                      # number of cores
#SBATCH --mem=128G                # memory pool for all cores
#SBATCH -t 6-23:59                # time (D-HH:MM)
#SBATCH --gres=gpu:1              # number of GPU
#SBATCH --job-name=main
#SBATCH -o slurm-main-output%a      # STDOUT
#SBATCH -e slurm-main-error%a       # STDERR

################################
# Run experiments in main.py (by submitting a slurm job) 
# How to use this script?
# in Cluster Head Node terminal, type: sbatch --array=1-5 main.sbatch 
# here the ids are the task ids to run 
################################

source ../env/bin/activate
python main.py --taskid=${SLURM_ARRAY_TASK_ID} --num_gpu=0 

################################
# How to use more GPUs?
# change line 6 and line 19
################################
