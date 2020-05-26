#!/bin/bash
################################
# Script for running multiple experiments by submitting multiple slurm jobs
# How to use this script?
#  ./main.sh [taskid range]
# For example, if you want to run task id 1 to id 5 (5 tasks in parallel), you could type:
# ./main.sh 1-5
################################
sbatch --array=$1 main.sbatch



################################
# How to use more GPUs?
# change in file: main.sbatch
################################
