#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=12:20:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
export WANDB_API_KEY=a67d6929d5168a0c365f4e73fe150ec7302e2824
export CUDA_LAUNCH_BLOCKING=1
python /home/partha9/SimLoc/simloc/report/finetune/DuplicateDetection.py --config_path /home/partha9/SimLoc/simloc/TrainingArgs/finetune/7.json --save_path /project/def-m2nagapp/partha9/SimLoc/Output/finetune --data_path /project/def-m2nagapp/partha9/SimLoc/Data/Processed/
